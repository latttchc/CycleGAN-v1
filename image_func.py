import pathlib
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# データパスを指定
data_root = pathlib.Path("datasets/horse2zebra")
train_x_paths = list(data_root.glob('trainA/*'))
train_x_paths = [str(path) for path in train_x_paths][:100]  # 最初の100個の画像を使用

train_y_paths = list(data_root.glob('trainB/*'))
train_y_paths = [str(path) for path in train_y_paths][:100]  # 最初の100個の画像を使用

test_x_paths = list(data_root.glob('testA/*'))
test_x_paths = [str(path) for path in test_x_paths][:100]  # 最初の100個の画像を使用

test_y_paths = list(data_root.glob('testB/*'))
test_y_paths = [str(path) for path in test_y_paths][:100]  # 最初の100個の画像を使用

# 訓練用の写真をクロッピング,左右反転,正規化するために変換
def get_train_transform():
    return transforms.Compose([
        transforms.Resize(286, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5 ], std=[0.5, 0.5, 0.5])
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5 ], std=[0.5, 0.5, 0.5])
    ])

# カスタムデータセットクラス
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
# データセットの作成
train_x_dataset = ImageDataset(train_x_paths, transform=get_train_transform())
train_y_dataset = ImageDataset(train_y_paths, transform=get_train_transform())
test_x_dataset = ImageDataset(test_x_paths, transform=get_test_transform())
test_y_dataset = ImageDataset(test_y_paths, transform=get_test_transform())

# DataLoaderの作成
train_x = DataLoader(train_x_dataset, batch_size=1, shuffle=True)
train_y = DataLoader(train_y_dataset, batch_size=1, shuffle=True)
test_x = DataLoader(test_x_dataset, batch_size=1, shuffle=False)
test_y = DataLoader(test_y_dataset, batch_size=1, shuffle=False)

# 各データの出力出力数を変数に格納
len_train_x = len(train_x_dataset)
len_train_y = len(train_y_dataset)
len_test_x = len(test_x_dataset)
len_test_y = len(test_y_dataset)

# 適当な一組の画像ペアを取り出す
sample_x = next(iter(train_x))
sample_y = next(iter(train_y))

# 正規化を元に戻す関数
def denormalize(tensor):
    return tensor * 0.5 + 0.5

# 左右反転用の変換
flip_transform = transforms.RandomHorizontalFlip(p=1.0)

# 元の画像と左右反転画像を表示
plt.figure(figsize=(12,8))

plt.subplot(221)
plt.title('Train A')
img_x = denormalize(sample_x[0]).permute(1, 2, 0).numpy()
plt.imshow(np.clip(img_x, 0, 1))
plt.axis('off')

plt.subplot(222)
plt.title('Drawing with random jitter')
flipped_x = flip_transform(sample_x[0])
img_flipped = denormalize(flipped_x).permute(1, 2, 0).numpy()
plt.imshow(np.clip(img_flipped, 0, 1))
plt.axis('off')

plt.subplot(223)
plt.title('Train B')
img_y = denormalize(sample_y[0]).permute(1, 2, 0).numpy()
plt.imshow(np.clip(img_y, 0, 1))
plt.axis('off')

plt.subplot(224)
plt.title('Drawing with random jitter')
flipped_y = flip_transform(sample_y[0])
img_y_flipped = denormalize(flipped_y).permute(1, 2, 0).numpy()
plt.imshow(np.clip(img_y_flipped, 0, 1))
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Train X dataset size: {len_train_x}")
print(f"Train Y dataset size: {len_train_y}")
print(f"Test X dataset size: {len_test_x}")
print(f"Test Y dataset size: {len_test_y}")
print(f"Sample X shape: {sample_x.shape}")
print(f"Sample Y shape: {sample_y.shape}")
