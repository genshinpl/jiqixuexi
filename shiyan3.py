# =============================================================================
# 2026task3 Fovea Localization - Full Solution
# ResNet34 + FPN + CenterNet-style Gaussian Heatmap
# =============================================================================

import os
import re
import math
import time
import copy
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
try:
    from torchvision.models import resnet34, ResNet34_Weights
except ImportError:
    from torchvision.models import resnet34
    ResNet34_Weights = None
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
cv2.setNumThreads(2)
IMREAD_FLAGS = cv2.IMREAD_COLOR | getattr(cv2, "IMREAD_IGNORE_ORIENTATION", 0)

CFG = {
    "data_root": os.environ.get(
        "FOVEA_DATA_ROOT",
        "/kaggle/input/2026task3/detection",
    ),
    "img_size": 768,
    "heat_sigma": 1.8,
    "folds": 5,
    "epochs": 40,
    "batch_size": 8,
    "infer_batch": 4,
    "lr": 2.5e-4,
    "backbone_lr_mult": 0.25,
    "weight_decay": 1e-4,
    "warmup_epochs": 3,
    "patience": 10,
    "seed": 42,
    "num_workers": 4,
}
CFG["hm_size"] = CFG["img_size"] // 4
SEED = CFG["seed"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("output")
WORK_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = WORK_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


set_seed(SEED)
print("torch:", torch.__version__, "| device:", DEVICE, "| GPUs:", torch.cuda.device_count())


def find_root():
    candidates = [
        Path(CFG["data_root"]),
        Path("/kaggle/input/2026task3/detection"),
        Path("/kaggle/input/competitions/2026task3/detection"),
        Path("/kaggle/input/2026task3"),
    ]
    for p in candidates:
        if (p / "train").is_dir() and (p / "test").is_dir():
            return p

    base = Path("/kaggle/input")
    hits = []
    if base.exists():
        hits = [p for p in base.rglob("*.csv")
                if p.is_file() and p.name.lower() == "fovea_localization_train_gt.csv"]
    if hits:
        return hits[0].parent

    print("CFG['data_root'] =", CFG["data_root"])
    print("/kaggle/input 顶层：")
    if base.exists():
        for p in sorted(base.iterdir())[:40]:
            print("  ", p)
    raise FileNotFoundError(
        "找不到数据。请先在 Notebook 右侧 Add Input → Competition Data 挂 2026task3；"
        "或者从比赛页 Code → New Notebook 新建（会自动挂数据）。"
    )


DATA_ROOT = find_root()
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
GT_CSV = DATA_ROOT / "fovea_localization_train_GT.csv"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"
if not SAMPLE_SUB.exists():
    hits = list(DATA_ROOT.rglob("sample_submission.csv"))
    if hits:
        SAMPLE_SUB = hits[0]

print("DATA_ROOT =", DATA_ROOT)
for p in [TRAIN_DIR, TEST_DIR, GT_CSV, SAMPLE_SUB]:
    print("  OK " if p.exists() else "  MISS ", p)


def canon_id(v):
    s = Path(str(v).strip()).stem
    s = re.sub(r"(?i)(?:_fovea_[xy]|_fovea|_x|_y)$", "", s).strip()
    return str(int(s)) if s.isdigit() else s.lower()


def index_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    out = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in exts:
            out[canon_id(p.stem)] = p
    return out


def image_size(path):
    with Image.open(path) as im:
        return int(im.width), int(im.height)


def _find_col(df, patterns, exclude=()):
    for c in df.columns:
        lc = str(c).lower()
        if any(re.search(p, lc) for p in patterns) and not any(re.search(e, lc) for e in exclude):
            return c
    return None


def load_gt(path):
    raw = pd.read_csv(path)
    print("GT columns:", list(raw.columns))
    print(raw.head(5).to_string())

    id_col = _find_col(raw, [r"image", r"img", r"file", r"name", r"^id$"])
    if id_col is None:
        id_col = raw.columns[0]
    x_col = _find_col(raw, [r"fovea.*x", r"(^|[_\s])x($|[_\s])", r"^x$"],
                      exclude=[r"image", r"img", r"file", r"name"])
    y_col = _find_col(raw, [r"fovea.*y", r"(^|[_\s])y($|[_\s])", r"^y$"],
                      exclude=[r"image", r"img", r"file", r"name"])

    if x_col is None or y_col is None:
        val_col = _find_col(raw, [r"^value$", r"label", r"coord"]) or raw.columns[-1]
        tmp = raw[[id_col, val_col]].copy()
        tmp.columns = ["raw_id", "value"]
        tmp["id"] = tmp["raw_id"].map(canon_id)
        tmp["axis"] = tmp["raw_id"].astype(str).str.extract(r"(?i)_([xy])$")[0].str.lower()
        wide = tmp.pivot_table(index="id", columns="axis", values="value", aggfunc="first").reset_index()
        wide.columns.name = None
        out = wide[["id", "x", "y"]].copy()
    else:
        out = raw[[id_col, x_col, y_col]].copy()
        out.columns = ["raw_id", "x", "y"]
        out["id"] = out["raw_id"].map(canon_id)
        out = out[["id", "x", "y"]]

    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    return out.dropna(subset=["id"]).drop_duplicates("id")


gt = load_gt(GT_CSV)
train_index = index_images(TRAIN_DIR)
test_index = index_images(TEST_DIR)
print("train images:", len(train_index), "| test images:", len(test_index))

rows = []
for k, p in train_index.items():
    rec = gt[gt["id"] == k]
    if len(rec) == 0:
        continue
    x, y = float(rec.iloc[0]["x"]), float(rec.iloc[0]["y"])
    if np.isnan(x) or np.isnan(y):
        x = y = 0.0
    w, h = image_size(p)
    rows.append(dict(id=k, image_path=str(p), x=x, y=y, w=w, h=h,
                     visible=float((x > 0) or (y > 0))))

labels = pd.DataFrame(rows)
if len(labels) == 0:
    raise RuntimeError("训练图像和 GT 没有匹配上，请检查文件名/ID。")

if labels["x"].max() <= 1.5 and labels["y"].max() <= 1.5:
    print("检测到 0~1 归一化坐标，转成像素")
    labels["x"] *= labels["w"]
    labels["y"] *= labels["h"]
elif labels["x"].max() <= 100.5 and labels["y"].max() <= 100.5 and labels["w"].min() > 200:
    print("检测到 0~100 百分比坐标，转成像素")
    labels["x"] *= labels["w"] / 100.0
    labels["y"] *= labels["h"] / 100.0

bad = (labels["x"] < 0) | (labels["y"] < 0) | (labels["x"] > labels["w"] + 2) | (labels["y"] > labels["h"] + 2)
labels.loc[bad, ["x", "y", "visible"]] = [0.0, 0.0, 0.0]
labels["x"] = labels["x"].clip(0, labels["w"] - 1)
labels["y"] = labels["y"].clip(0, labels["h"] - 1)

N = len(labels)
N_INVIS = int((labels["visible"] < 0.5).sum())
print("matched:", N, "| visible:", N - N_INVIS, "| invisible:", N_INVIS)
print("size:", int(labels.w.min()), int(labels.w.max()), "x", int(labels.h.min()), int(labels.h.max()))
labels.to_csv(WORK_DIR / "train_labels.csv", index=False)


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox(img, size):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, scale, px, py


def unletterbox(xn, yn, w, h, size):
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    px, py = (size - nw) // 2, (size - nh) // 2
    x = (xn * size - px) / max(scale, 1e-8)
    y = (yn * size - py) / max(scale, 1e-8)
    return float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))


def make_heatmap(xn, yn, hm_size, sigma, visible):
    hm = np.zeros((hm_size, hm_size), dtype=np.float32)
    if visible <= 0:
        return hm
    cx = float(np.clip(xn * hm_size, 0, hm_size - 1))
    cy = float(np.clip(yn * hm_size, 0, hm_size - 1))
    ix, iy = int(round(cx)), int(round(cy))
    hm[iy, ix] = 1.0
    radius = max(1, int(3 * sigma))
    x0, x1 = max(0, ix - radius), min(hm_size, ix + radius + 1)
    y0, y1 = max(0, iy - radius), min(hm_size, iy + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g)
    return hm


def color_augment(img):
    img = img.astype(np.float32)
    if random.random() < 0.8:
        img = img * random.uniform(0.85, 1.15) + random.uniform(-12, 12)
    if random.random() < 0.3:
        img = 255.0 * np.power(np.clip(img, 0, 255) / 255.0, random.uniform(0.8, 1.25))
    if random.random() < 0.2:
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.4, 1.3))
    return np.clip(img, 0, 255).astype(np.uint8)


class FoveaDataset(Dataset):
    def __init__(self, df, train):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.S = CFG["img_size"]
        self.HM = CFG["hm_size"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(r.image_path, IMREAD_FLAGS)
        if img is None:
            raise FileNotFoundError(r.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, scale, px, py = letterbox(img, self.S)
        visible = float(r.visible)
        x, y = float(r.x) * scale + px, float(r.y) * scale + py

        if self.train:
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
                if visible > 0:
                    x = self.S - 1 - x
            angle = random.uniform(-12, 12)
            sc = random.uniform(0.86, 1.14)
            M = cv2.getRotationMatrix2D((self.S / 2, self.S / 2), angle, sc)
            M[0, 2] += random.uniform(-0.06, 0.06) * self.S
            M[1, 2] += random.uniform(-0.06, 0.06) * self.S
            nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
            ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
            if visible <= 0 or (2 <= nx < self.S - 2 and 2 <= ny < self.S - 2):
                img = cv2.warpAffine(img, M, (self.S, self.S),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                if visible > 0:
                    x, y = nx, ny
            img = color_augment(img)

        xn = float(np.clip(x / self.S, 0, 1))
        yn = float(np.clip(y / self.S, 0, 1))
        heat = make_heatmap(xn, yn, self.HM, CFG["heat_sigma"], visible)
        arr = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return {
            "img": torch.from_numpy(arr.transpose(2, 0, 1)),
            "heat": torch.from_numpy(heat).unsqueeze(0),
            "vis": torch.tensor(visible, dtype=torch.float32),
            "xy": torch.tensor([xn, yn], dtype=torch.float32),
        }


def make_loader(df, train, batch_size, shuffle):
    return DataLoader(
        FoveaDataset(df, train), batch_size=batch_size, shuffle=shuffle,
        num_workers=CFG["num_workers"], pin_memory=True,
        persistent_workers=CFG["num_workers"] > 0, drop_last=False,
    )


def conv_bn(in_ch, out_ch, k=3):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True),
    )


class FoveaNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = None
        if pretrained and ResNet34_Weights is not None:
            try:
                weights = ResNet34_Weights.IMAGENET1K_V1
            except Exception:
                weights = None
        try:
            bb = resnet34(weights=weights)
        except TypeError:
            bb = resnet34(pretrained=weights is not None)
        except Exception as e:
            print("预训练权重加载失败，随机初始化：", e)
            bb = resnet34(weights=None)

        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1, self.layer2 = bb.layer1, bb.layer2
        self.layer3, self.layer4 = bb.layer3, bb.layer4
        c2, c3, c4, c5, feat = 64, 128, 256, 512, 128
        self.lat5 = nn.Conv2d(c5, feat, 1)
        self.lat4 = nn.Conv2d(c4, feat, 1)
        self.lat3 = nn.Conv2d(c3, feat, 1)
        self.lat2 = nn.Conv2d(c2, feat, 1)
        self.smooth = conv_bn(feat, feat)
        self.drop = nn.Dropout2d(0.10)
        self.heat = nn.Sequential(conv_bn(feat, 64), nn.Conv2d(64, 1, 1))
        self.vis = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c5, 1))
        nn.init.constant_(self.heat[-1].bias, -4.5)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        feat = self.drop(self.smooth(p2))
        return self.heat(feat), self.vis(c5).squeeze(1)


def focal_loss(logits, target):
    pred = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    pos = (target >= 1 - 1e-4).float()
    neg = 1 - pos
    pos_loss = -torch.log(pred) * (1 - pred) ** 2 * pos
    neg_loss = -torch.log(1 - pred) * pred ** 2 * (1 - target) ** 4 * neg
    num_pos = pos.sum(dim=(1, 2, 3))
    denom = torch.where(num_pos > 0, num_pos, neg.sum(dim=(1, 2, 3)).clamp_min(1.0))
    return ((pos_loss + neg_loss).sum(dim=(1, 2, 3)) / denom).mean()


def decode_heatmap(heat, radius=7):
    arr = heat[:, 0].detach().float().cpu().numpy()
    B, H, W = arr.shape
    xy = np.zeros((B, 2), dtype=np.float32)
    conf = np.zeros((B,), dtype=np.float32)
    for b in range(B):
        hm = arr[b]
        py, px = np.unravel_index(int(np.argmax(hm)), (H, W))
        conf[b] = float(hm[py, px])
        y0, y1 = max(0, py - radius), min(H, py + radius + 1)
        x0, x1 = max(0, px - radius), min(W, px + radius + 1)
        win = hm[y0:y1, x0:x1].astype(np.float64)
        if win.sum() <= 1e-8:
            xy[b] = [(px + 0.5) / W, (py + 0.5) / H]
            continue
        w = win ** 2
        ys, xs = np.mgrid[y0:y1, x0:x1]
        xy[b] = [float((w * (xs + 0.5)).sum() / w.sum()) / W,
                 float((w * (ys + 0.5)).sum() / w.sum()) / H]
    return xy, conf


@torch.no_grad()
def predict(model, loader, tta=True):
    model.eval()
    P = {"xy": [], "conf": [], "vis": [], "heat": []}
    for batch in loader:
        imgs = batch["img"].to(DEVICE, non_blocking=True)
        if tta:
            out_heat, out_vis = model(torch.cat([imgs, torch.flip(imgs, dims=[3])], 0))
            B = imgs.size(0)
            heat = 0.5 * (torch.sigmoid(out_heat[:B]) +
                          torch.flip(torch.sigmoid(out_heat[B:]), dims=[3]))
            vis = 0.5 * (torch.sigmoid(out_vis[:B]) + torch.sigmoid(out_vis[B:]))
        else:
            out_heat, out_vis = model(imgs)
            heat = torch.sigmoid(out_heat)
            vis = torch.sigmoid(out_vis)
        xy, conf = decode_heatmap(heat)
        P["xy"].append(xy)
        P["conf"].append(conf)
        P["vis"].append(vis.detach().cpu().numpy())
        P["heat"].append(heat[:, 0].detach().float().cpu().numpy().astype(np.float16))
    return {k: np.concatenate(v, 0) for k, v in P.items()}


def rmse(pred, gt):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(gt)) ** 2)))


def train_fold(fold, tr_df, va_df):
    print(f"\n===== FOLD {fold} train={len(tr_df)} val={len(va_df)} =====")
    tr_loader = make_loader(tr_df, True, CFG["batch_size"], True)
    va_loader = make_loader(va_df, False, CFG["infer_batch"], False)
    model = FoveaNet(pretrained=True).to(DEVICE)

    bb = [p for n, p in model.named_parameters() if n.startswith(("stem", "layer"))]
    head = [p for n, p in model.named_parameters() if not n.startswith(("stem", "layer"))]
    opt = AdamW([
        {"params": bb, "lr": CFG["lr"] * CFG["backbone_lr_mult"]},
        {"params": head, "lr": CFG["lr"]},
    ], weight_decay=CFG["weight_decay"])

    def lr_lambda(epoch):
        w = CFG["warmup_epochs"]
        if epoch < w:
            return (epoch + 1) / w
        progress = (epoch - w) / max(1, CFG["epochs"] - w)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best = 1e18
    best_state = None
    bad = 0

    for epoch in range(CFG["epochs"]):
        model.train()
        losses = []
        t0 = time.time()
        for batch in tr_loader:
            imgs = batch["img"].to(DEVICE, non_blocking=True)
            heat_t = batch["heat"].to(DEVICE, non_blocking=True)
            vis_t = batch["vis"].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                heat, vis = model(imgs)
                loss = focal_loss(heat, heat_t)
                if N_INVIS > 0:
                    loss = loss + 0.2 * F.binary_cross_entropy_with_logits(vis, vis_t)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        sched.step()

        p = predict(model, va_loader, tta=False)
        gt = va_df[["x", "y"]].values.astype(np.float64)
        pred = np.zeros_like(gt)
        for i, r in va_df.reset_index(drop=True).iterrows():
            if r.visible > 0:
                pred[i] = unletterbox(p["xy"][i, 0], p["xy"][i, 1], int(r.w), int(r.h), CFG["img_size"])
        mask = va_df["visible"].values > 0.5
        rv = rmse(pred[mask], gt[mask]) if mask.any() else 1e9
        print(f"epoch {epoch+1:02d}/{CFG['epochs']} loss={np.mean(losses):.4f} "
              f"RMSE_vis={rv:.2f} ({time.time()-t0:.1f}s)")

        if rv < best - 1e-6:
            best = rv
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, CKPT_DIR / f"fold{fold}.pt")
            bad = 0
        else:
            bad += 1
            if bad >= CFG["patience"]:
                print("early stop")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


if N_INVIS > 0 and labels["visible"].value_counts().min() >= CFG["folds"]:
    splitter = StratifiedKFold(n_splits=CFG["folds"], shuffle=True, random_state=SEED)
    splits = list(splitter.split(labels, labels["visible"].astype(int)))
else:
    splitter = KFold(n_splits=CFG["folds"], shuffle=True, random_state=SEED)
    splits = list(splitter.split(labels))

models = []
for fold, (tr_idx, va_idx) in enumerate(splits):
    model = train_fold(fold, labels.iloc[tr_idx].copy(), labels.iloc[va_idx].copy())
    models.append(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# OOF inference + threshold search
# =============================================================================

oof_xy = []
oof_vis = []
oof_gt = []
oof_visible = []

for fold, (tr_idx, va_idx) in enumerate(splits):
    va_df = labels.iloc[va_idx].copy().reset_index(drop=True)
    loader = make_loader(va_df, False, CFG["infer_batch"], False)
    p = predict(models[fold], loader, tta=True)
    for i, r in va_df.iterrows():
        if r.visible > 0:
            xy = unletterbox(p["xy"][i, 0], p["xy"][i, 1], int(r.w), int(r.h), CFG["img_size"])
        else:
            xy = (0.0, 0.0)
        oof_xy.append(xy)
        oof_vis.append(float(p["vis"][i]))
        oof_gt.append((float(r.x), float(r.y)))
        oof_visible.append(float(r.visible))

oof_xy = np.asarray(oof_xy)
oof_vis = np.asarray(oof_vis)
oof_gt = np.asarray(oof_gt)
oof_visible = np.asarray(oof_visible)

mask = oof_visible > 0.5
print("OOF visible RMSE:", rmse(oof_xy[mask], oof_gt[mask]))

best_thr = 0.0
if N_INVIS > 0:
    best_rmse = 1e18
    for thr in np.arange(0.05, 0.96, 0.05):
        pred = oof_xy.copy()
        pred[oof_vis < thr] = 0.0
        r = rmse(pred, oof_gt)
        if r < best_rmse:
            best_rmse = r
            best_thr = float(thr)
    print("best threshold:", best_thr, "OOF all RMSE:", best_rmse)
else:
    print("训练集没有不可见样本，threshold=0，全部输出坐标")


# =============================================================================
# Test inference + submission
# =============================================================================

test_df = pd.DataFrame([dict(id=k, image_path=str(p)) for k, p in test_index.items()])
for i, r in test_df.iterrows():
    w, h = image_size(Path(r.image_path))
    test_df.at[i, "w"] = w
    test_df.at[i, "h"] = h
for c in ["x", "y", "visible"]:
    test_df[c] = 0.0

test_loader = make_loader(test_df, False, CFG["infer_batch"], False)
heat_sum = None
vis_list = []

for fold, model in enumerate(models):
    p = predict(model, test_loader, tta=True)
    hf = p["heat"].astype(np.float32)
    heat_sum = hf if heat_sum is None else heat_sum + hf
    vis_list.append(p["vis"])
    print("test fold", fold, "done")

heat_mean = torch.from_numpy(heat_sum / len(models)).unsqueeze(1)
xy_norm, _ = decode_heatmap(heat_mean)
vis_prob = np.median(np.stack(vis_list, 0), axis=0)

coords = np.zeros((len(test_df), 2), dtype=np.float64)
for i, r in test_df.iterrows():
    coords[i] = unletterbox(xy_norm[i, 0], xy_norm[i, 1], int(r.w), int(r.h), CFG["img_size"])

if best_thr > 0:
    coords[vis_prob < best_thr] = 0.0

sample = pd.read_csv(SAMPLE_SUB)
pred_map = {str(r.id): (float(coords[i, 0]), float(coords[i, 1]))
            for i, r in test_df.iterrows()}

values = []
for raw in sample.iloc[:, 0].astype(str):
    m = re.match(r"(.+?)_Fovea_([XY])$", raw.strip(), flags=re.I)
    if not m:
        values.append(0.0)
        continue
    img_id = canon_id(m.group(1))
    axis = m.group(2).upper()
    values.append(pred_map.get(img_id, (0.0, 0.0))[0 if axis == "X" else 1])

sample.iloc[:, 1] = values
out_path = WORK_DIR / "submission.csv"
sample.to_csv(out_path, index=False)
print("saved:", out_path)
print(sample.head(12).to_string(index=False))
print("zeros:", int((coords.sum(axis=1) == 0).sum()), "| threshold:", best_thr)