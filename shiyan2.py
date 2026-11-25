import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd
import numpy as np

# ================= 1. 配置参数 =================
# 注意这里已经加上了 /train
DATA_DIR = r"/kaggle/input/competitions/2026task2/fer_data/fer_data/train"
TEST_DIR = r"/kaggle/input/competitions/2026task2/fer_data/fer_data/test"
OUTPUT_DIR = '/kaggle/working'

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 48
PATCH_SIZE = 6
IN_CHANNELS = 1
NUM_CLASSES = 6
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_TO_IDX = {
    "Angry": 0, "Fear": 1, "Happy": 2,
    "Sad": 3, "Surprise": 4, "Neutral": 5
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class CustomImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: CLASS_TO_IDX[cls_name] for cls_name in classes if cls_name in CLASS_TO_IDX}
        return classes, class_to_idx


class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_names = sorted(os.listdir(test_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, img_name



full_dataset = CustomImageFolder(root=DATA_DIR, transform=train_transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=6, embed_dim=128, img_size=48):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class SimpleViT(nn.Module):
    def __init__(self, img_size=48, patch_size=6, in_channels=1, num_classes=6,
                 embed_dim=128, depth=4, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, img_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)

        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)
        out = self.head(cls_output)
        return out


model = SimpleViT(
    img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=IN_CHANNELS,
    num_classes=NUM_CLASSES, embed_dim=128, depth=4, num_heads=4
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

print("开始训练模型...")
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_vit_fer.pth'))

print(f"训练结束，最高验证集准确率: {best_acc:.2f}%")

print("开始对测试集进行推理...")
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_vit_fer.pth')))
model.eval()

test_dataset = TestDataset(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

predictions = []
image_names_list = []

with torch.no_grad():
    for images, img_names in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        predictions.extend(predicted.cpu().numpy())
        image_names_list.extend(img_names)

submission_df = pd.DataFrame({
    'image_id': image_names_list,
    'label': predictions
})

submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
submission_df.to_csv(submission_path, index=False)
print(f"预测完成，结果已保存至: {submission_path}")