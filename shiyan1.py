import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

# ===================== 【改动】更强训练增强，加入色彩抖动、随机擦除 =====================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 验证、测试集不用增强
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. 填入你确认的准确嵌套路径
TRAIN_DIR = '/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2/train'
TEST_DIR = '/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2/test'

# 3. 加载训练集，【改动】划分训练集 / 验证集 8:2
try:
    full_train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    # 验证集关闭增强
    val_dataset.dataset.transform = val_test_transforms

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    print("✅ 成功加载训练+验证集！")
    print("检测到的植物类别:", full_train_dataset.classes)
    print("类别映射字典:", full_train_dataset.class_to_idx)
    print(f"训练集图片数量: {len(train_dataset)}")
    print(f"验证集图片数量: {len(val_dataset)}")
except Exception as e:
    print(f"❌ 加载训练集失败: {e}")


# 4. 加载测试集
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, img_name


try:
    test_dataset = TestDataset(TEST_DIR, transform=val_test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"✅ 成功加载测试集！测试集图片数量: {len(test_dataset)}")
except Exception as e:
    print(f"❌ 加载测试集失败: {e}")