import pathlib
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

# データパスを指定
data_root = pathlib.Path("datasets/horse2zebra")
train_x_paths = list(data_root.glob('trainA/*'))
train_x_paths = [str(path) for path in train_x_paths ]

train_y_paths = list(data_root.glob('trainB/*'))
train_y_paths = [str(path) for path in train_y_paths ]

test_x_paths = list(data_root.glob('testA/*'))
test_x_paths = [str(path) for path in test_x_paths ]

test_y_paths = list(data_root.glob('testB/*'))
test_y_paths = [str(path) for path in test_y_paths ]

# ランダムに画像を選択
painting_path = random.choice(train_x_paths)
photo_path = random.choice(train_y_paths)

# 画像変換を定義
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 画像を読み込み
painting = Image.open(painting_path).convert('RGB')
photo = Image.open(photo_path).convert('RGB')

# pytorchテンソルに変換
painting_tensor = transform(painting)
photo_tensor = transform(photo)

# バッチ次元を追加
painting_np = painting_tensor.permute(1, 2, 0).numpy()
photo_np = photo_tensor.permute(1, 2, 0).numpy()

# 画像を表示
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax1 = axes[0]
ax1.imshow(painting_np)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Train A')

ax2 = axes[1]
ax2.imshow(photo_np)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title('Train B')

plt.show()

print(f"Painting tensor shape: {painting_tensor.shape}")
print(f"Photo tensor shape: {photo_tensor.shape}")
print(f"Selected painting: {painting_path}")
print(f"Selected photo: {photo_path}")
