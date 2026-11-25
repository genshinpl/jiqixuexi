import os, sys, glob, math, time, random, shutil, subprocess, warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings('ignore')
cv2.setNumThreads(4)
SEED        = 42
N_FOLDS     = 5
ITERS       = 4000
EVAL_EVERY  = 400
BATCH       = 8
PATCH       = 256
BASE        = 32
LR          = 3e-4
WD          = 1e-4
EMA_DECAY   = 0.999
STRIDE      = 128
USE_TTA     = True
MIN_AREA    = 20
INVERT_MASK = False
WORK        = '/kaggle/working'

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

# ------------------------------- 路径 ----------------------------------------
def find_root():
    cands = ['/kaggle/input/competitions/2026task-4/segmentation',
             '/kaggle/input/2026task-4/segmentation',
             '/kaggle/input/segmentation',
             '/kaggle/input/competitions/2026task-4',
             'segmentation', './segmentation', '../segmentation']
    for c in cands:
        if os.path.isdir(os.path.join(c, 'train', 'image')):
            return os.path.abspath(c)
    hits = glob.glob('/kaggle/input/**/train/image', recursive=True)
    if hits:
        return os.path.abspath(os.path.dirname(os.path.dirname(hits[0])))
    raise RuntimeError('找不到数据目录')

ROOT = find_root()
os.makedirs(WORK, exist_ok=True)
print('DEV =', DEV, '| ROOT =', ROOT)

# ------------------------------- 工具 ----------------------------------------
def seed_all(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def list_ids(d):
    exts = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')
    fs = [f for f in os.listdir(d) if f.lower().endswith(exts)]
    def key(f):
        s = os.path.splitext(f)[0]
        return (0, int(s)) if s.isdigit() else (1, s)
    return sorted([os.path.splitext(f)[0] for f in fs], key=key)

def fpath(d, i):
    for ext in ('.jpg', '.jpeg', '.png', '.tif', '.bmp'):
        p = os.path.join(d, i + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(os.path.join(d, i))

def read_rgb(p):
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(p)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def read_mask(p):
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(p)
    return (m > 127).astype(np.uint8) * 255

def make_feat(rgb):
    """4 通道输入: 归一化 RGB + 黑帽滤波(增强暗色细线)"""
    f = rgb.astype(np.float32) / 255.0
    g = f.mean(2)
    mu, sd = float(g.mean()), float(g.std() + 1e-6)
    x = (f - mu) / sd
    g8 = np.clip(g * 255.0, 0, 255).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    th = cv2.morphologyEx(g8, cv2.MORPH_BLACKHAT, ker).astype(np.float32) / 255.0
    th = th / (float(th.std()) + 1e-6)
    return np.concatenate([x, th[..., None]], 2).astype(np.float32)

def dice_np(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    s = pred.sum() + gt.sum()
    if s == 0:
        return 1.0
    return float(2.0 * (pred & gt).sum() / s)

def postprocess(mask, min_area=MIN_AREA):
    mask = mask.astype(np.uint8)
    if min_area <= 1:
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = mask.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            out[lab == i] = 0
    return out

# ------------------------------- 读数据 --------------------------------------
TRAIN_IDS = list_ids(os.path.join(ROOT, 'train', 'image'))
TEST_IDS  = list_ids(os.path.join(ROOT, 'test', 'image'))
print('train:', len(TRAIN_IDS), 'test:', len(TEST_IDS))

print('load train ...')
XTR, YTR = [], []
for i in TRAIN_IDS:
    rgb = read_rgb(fpath(os.path.join(ROOT, 'train', 'image'), i))
    msk = read_mask(fpath(os.path.join(ROOT, 'train', 'label'), i))
    assert rgb.shape[:2] == msk.shape[:2], i
    XTR.append(make_feat(rgb))
    YTR.append(msk)

print('load test ...')
XTE, TEST_SIZE = [], []
for i in TEST_IDS:
    rgb = read_rgb(fpath(os.path.join(ROOT, 'test', 'image'), i))
    XTE.append(make_feat(rgb))
    TEST_SIZE.append(rgb.shape[:2])

VESSEL_PTS = []
for y in YTR:
    ys, xs = np.nonzero(y > 0)
    VESSEL_PTS.append(np.stack([xs, ys], 1) if len(xs) else np.zeros((0, 2), np.int64))

# ------------------------------- 模型 ----------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.b(x)

class UNet(nn.Module):
    def __init__(self, cin=4, base=BASE):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.pool = nn.MaxPool2d(2)
        self.e1 = DoubleConv(cin, c1)
        self.e2 = DoubleConv(c1, c2)
        self.e3 = DoubleConv(c2, c3)
        self.e4 = DoubleConv(c3, c4)
        self.bt = DoubleConv(c4, c5)
        self.d4 = DoubleConv(c5 + c4, c4)
        self.d3 = DoubleConv(c4 + c3, c3)
        self.d2 = DoubleConv(c3 + c2, c2)
        self.d1 = DoubleConv(c2 + c1, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def up(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bt(self.pool(e4))
        d4 = self.d4(torch.cat([self.up(b, e4), e4], 1))
        d3 = self.d3(torch.cat([self.up(d4, e3), e3], 1))
        d2 = self.d2(torch.cat([self.up(d3, e2), e2], 1))
        d1 = self.d1(torch.cat([self.up(d2, e1), e1], 1))
        return self.head(d1)

# ------------------------------- 损失 / EMA ----------------------------------
bce = nn.BCEWithLogitsLoss()

def loss_fn(logit, target):
    l = bce(logit, target)
    p = torch.sigmoid(logit)
    num = 2.0 * (p * target).sum((2, 3)) + 1.0
    den = p.sum((2, 3)) + target.sum((2, 3)) + 1.0
    return l + (1.0 - (num / den)).mean()

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                self.shadow[k] = v.detach().clone()
    def copy_to(self, model):
        model.load_state_dict({k: v.clone() for k, v in self.shadow.items()})

# ------------------------------- 采样 ----------------------------------------
PAD = PATCH // 2

def sample_one(k):
    feat, mask = XTR[k], YTR[k]
    H, W = mask.shape
    ang = np.random.uniform(-180.0, 180.0)
    sc  = np.random.uniform(0.85, 1.30)
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, sc)
    M[0, 2] += PAD + np.random.uniform(-0.08, 0.08) * W
    M[1, 2] += PAD + np.random.uniform(-0.08, 0.08) * H
    size = (W + 2 * PAD, H + 2 * PAD)
    f = cv2.warpAffine(feat, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    m = cv2.warpAffine(mask, M, size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

    pts = VESSEL_PTS[k]
    if len(pts) and np.random.rand() < 0.6:
        x, y = pts[np.random.randint(len(pts))]
        cx = M[0, 0] * x + M[0, 1] * y + M[0, 2] + np.random.uniform(-PATCH / 8, PATCH / 8)
        cy = M[1, 0] * x + M[1, 1] * y + M[1, 2] + np.random.uniform(-PATCH / 8, PATCH / 8)
    else:
        cx = np.random.uniform(0, size[0])
        cy = np.random.uniform(0, size[1])

    x0 = int(np.clip(cx - PATCH / 2.0, 0, size[0] - PATCH))
    y0 = int(np.clip(cy - PATCH / 2.0, 0, size[1] - PATCH))
    f = f[y0:y0 + PATCH, x0:x0 + PATCH]
    m = m[y0:y0 + PATCH, x0:x0 + PATCH]

    if np.random.rand() < 0.5:
        f = np.ascontiguousarray(f[:, ::-1]); m = np.ascontiguousarray(m[:, ::-1])
    if np.random.rand() < 0.5:
        f = np.ascontiguousarray(f[::-1]);     m = np.ascontiguousarray(m[::-1])
    k90 = np.random.randint(4)
    if k90:
        f = np.ascontiguousarray(np.rot90(f, k90))
        m = np.ascontiguousarray(np.rot90(m, k90))

    p = f[..., :3]
    if np.random.rand() < 0.7:
        p = p * np.random.uniform(0.75, 1.30) + np.random.uniform(-0.25, 0.25)
    if np.random.rand() < 0.5:
        g = np.random.uniform(0.6, 1.6)
        p = np.sign(p) * np.abs(p) ** g
    if np.random.rand() < 0.3:
        p = p + np.random.randn(*p.shape).astype(np.float32) * np.random.uniform(0.02, 0.10)
    if np.random.rand() < 0.15:
        p = cv2.GaussianBlur(p, (3, 3), 0)
    f = np.concatenate([p, f[..., 3:]], 2)
    if np.random.rand() < 0.20:   # cutout
        cw = np.random.randint(8, 40); ch = np.random.randint(8, 40)
        xx = np.random.randint(0, PATCH - cw); yy = np.random.randint(0, PATCH - ch)
        f[yy:yy + ch, xx:xx + cw, :3] = np.random.uniform(-1.0, 1.0, (ch, cw, 3)).astype(np.float32)

    return np.ascontiguousarray(f), (m > 127).astype(np.float32)

def batch_tensors(items):
    x = np.stack([it[0] for it in items]).transpose(0, 3, 1, 2)
    y = np.stack([it[1] for it in items])[:, None]
    return torch.from_numpy(x).float(), torch.from_numpy(y).float()

# ------------------------------- 滑窗推理 ------------------------------------
@torch.no_grad()
def predict_prob(model, feat, tta=True):
    H, W = feat.shape[:2]
    ph, pw = (-H) % 16, (-W) % 16
    img = np.pad(feat, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    if img.shape[0] < PATCH or img.shape[1] < PATCH:
        ex = max(0, PATCH - img.shape[0]); ey = max(0, PATCH - img.shape[1])
        img = np.pad(img, ((0, ex), (0, ey), (0, 0)), mode='reflect')
    Hp, Wp = img.shape[:2]

    ys = list(range(0, Hp - PATCH + 1, STRIDE))
    if ys[-1] != Hp - PATCH:
        ys.append(Hp - PATCH)
    xs = list(range(0, Wp - PATCH + 1, STRIDE))
    if xs[-1] != Wp - PATCH:
        xs.append(Wp - PATCH)

    variants = [(False, False), (True, False), (False, True), (True, True)] if tta else [(False, False)]
    prob = np.zeros((Hp, Wp), np.float32)
    for fh, fv in variants:
        v = img
        if fh:
            v = np.flip(v, 1)
        if fv:
            v = np.flip(v, 0)
        v = np.ascontiguousarray(v)

        coord = [(y, x) for y in ys for x in xs]
        acc = np.zeros((Hp, Wp), np.float32)
        cnt = np.zeros((Hp, Wp), np.float32)
        for s in range(0, len(coord), 8):
            chunk = coord[s:s + 8]
            b = np.stack([v[y:y + PATCH, x:x + PATCH] for y, x in chunk]).transpose(0, 3, 1, 2)
            t = torch.from_numpy(b).float().to(DEV)
            with torch.autocast('cuda', enabled=(DEV == 'cuda')):
                o = torch.sigmoid(model(t)).float().cpu().numpy()[:, 0]
            for j, (y, x) in enumerate(chunk):
                acc[y:y + PATCH, x:x + PATCH] += o[j]
                cnt[y:y + PATCH, x:x + PATCH] += 1.0
        acc = acc / np.maximum(cnt, 1e-6)
        if fh:
            acc = np.flip(acc, 1)
        if fv:
            acc = np.flip(acc, 0)
        prob += acc
    prob /= float(len(variants))
    return prob[:H, :W]

# ------------------------------- 训练 ----------------------------------------
idx_all = np.arange(len(TRAIN_IDS))
np.random.RandomState(SEED).shuffle(idx_all)
FOLDS = [idx_all[i::N_FOLDS] for i in range(N_FOLDS)]

test_accum = {i: np.zeros(TEST_SIZE[t], np.float32) for t, i in enumerate(TEST_IDS)}
oof = {}
t0 = time.time()

for fold in range(N_FOLDS):
    val_idx = FOLDS[fold]
    tr_idx = np.setdiff1d(idx_all, val_idx)
    seed_all(SEED + fold)

    model = UNet(4, BASE).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEV == 'cuda'))
    ema = EMA(model, EMA_DECAY)
    best_d, best_state = -1.0, None

    for it in range(1, ITERS + 1):
        if it <= 200:
            lr = LR * it / 200.0
        else:
            tt = (it - 200) / float(max(1, ITERS - 200))
            lr = 1e-5 + 0.5 * (LR - 1e-5) * (1.0 + math.cos(math.pi * tt))
        for g in opt.param_groups:
            g['lr'] = lr

        model.train()
        items = [sample_one(int(tr_idx[np.random.randint(len(tr_idx))])) for _ in range(BATCH)]
        x, y = batch_tensors(items)
        x, y = x.to(DEV), y.to(DEV)
        with torch.cuda.amp.autocast(enabled=(DEV == 'cuda')):
            logit = model(x)
            loss = loss_fn(logit, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        ema.update(model)

        if it % EVAL_EVERY == 0 or it == ITERS:
            em = UNet(4, BASE).to(DEV)
            ema.copy_to(em); em.eval()
            ds = []
            for vi in val_idx:
                pv = predict_prob(em, XTR[vi], tta=False)
                ds.append(dice_np(pv > 0.5, YTR[vi] > 0))
            d = float(np.mean(ds))
            if d > best_d:
                best_d = d
                best_state = {k: v.detach().cpu().clone() for k, v in em.state_dict().items()}
            print('[fold%d] it %5d  loss %.4f  valDice %.4f  best %.4f  (%.1f min)'
                  % (fold, it, loss.item(), d, best_d, (time.time() - t0) / 60.0), flush=True)
            del em
            if DEV == 'cuda':
                torch.cuda.empty_cache()

    torch.save(best_state, os.path.join(WORK, 'unet_fold%d.pth' % fold))

    model.load_state_dict(best_state)
    model.eval()
    for vi in val_idx:
        oof[TRAIN_IDS[vi]] = predict_prob(model, XTR[vi], tta=USE_TTA)
    for ti, tid in enumerate(TEST_IDS):
        test_accum[tid] += predict_prob(model, XTE[ti], tta=USE_TTA) / float(N_FOLDS)

    del model
    if DEV == 'cuda':
        torch.cuda.empty_cache()

# ------------------------------- 阈值搜索 ------------------------------------
print('search threshold on OOF ...')
best_t, best_oof = 0.5, -1.0
for t in np.arange(0.20, 0.81, 0.02):
    ds = []
    for k, tid in enumerate(TRAIN_IDS):
        mk = postprocess((oof[tid] > t).astype(np.uint8))
        ds.append(dice_np(mk > 0, YTR[k] > 0))
    d = float(np.mean(ds))
    if d > best_oof:
        best_oof, best_t = d, float(t)
print('best OOF Dice = %.4f @ thr = %.2f' % (best_oof, best_t))

# ------------------------------- 写预测图 + 生成 csv -------------------------
OUTDIR = os.path.join(WORK, 'image')
shutil.rmtree(OUTDIR, ignore_errors=True)
os.makedirs(OUTDIR, exist_ok=True)

vessel_val = 0 if INVERT_MASK else 255
bg_val     = 255 if INVERT_MASK else 0
for tid in TEST_IDS:
    m = postprocess((test_accum[tid] > best_t).astype(np.uint8))
    arr = np.where(m > 0, vessel_val, bg_val).astype(np.uint8)
    Image.fromarray(arr, mode='L').save(os.path.join(OUTDIR, tid + '.png'))

def rle_encode(png_path):
    """与官方 segmentation_to_csv.py 完全一致: 转 L -> resize 512 -> >0 -> RLE"""
    im = Image.open(png_path).convert('L').resize((512, 512), Image.BILINEAR)
    a = np.asarray(im).astype(np.float32) / 255.0
    flat = (a.ravel() > 0)
    idx = np.nonzero(flat)[0]
    runs = []
    if len(idx):
        start = prev = int(idx[0])
        for b in idx[1:]:
            b = int(b)
            if b != prev + 1:
                runs += [start + 1, prev - start + 1]
                start = b
            prev = b
        runs += [start + 1, prev - start + 1]
    return ' '.join(str(v) for v in runs)

SUB = os.path.join(WORK, 'submission.csv')
seg_py = os.path.join(ROOT, 'segmentation_to_csv.py')
done = False
if os.path.exists(seg_py):
    try:
        print('run official segmentation_to_csv.py ...')
        subprocess.run([sys.executable, seg_py], cwd=WORK, check=True)
        done = os.path.exists(SUB)
    except Exception as e:
        print('official script failed:', e, '-> fallback inline encoder')
if not done:
    import pandas as pd
    pd.DataFrame({'Id': TEST_IDS,
                  'Predicted': [rle_encode(os.path.join(OUTDIR, i + '.png')) for i in TEST_IDS]}
                 ).to_csv(SUB, index=False)

# 排序 + 自检
import pandas as pd
df = pd.read_csv(SUB).fillna('')
df['_k'] = df['Id'].astype(str).str.extract(r'(\d+)').astype(int)
df = df.sort_values('_k').drop(columns='_k')
df = df[['Id', 'Predicted']]
df.to_csv(SUB, index=False)

assert len(df) == len(TEST_IDS), (len(df), len(TEST_IDS))
assert set(df['Id'].astype(str)) == set(TEST_IDS)
fracs = []
for s in df['Predicted'].astype(str):
    v = np.array(s.split(), dtype=np.int64) if s.strip() else np.zeros(0, np.int64)
    fracs.append(v.reshape(-1, 2)[:, 1].sum() / (512.0 * 512.0))
print('rows:', len(df), '| mean RLE-covered fraction: %.4f' % float(np.mean(fracs)))
print(df.head())
print('saved ->', SUB)