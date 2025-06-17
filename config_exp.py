import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # デバイスの設定
TRAIN_DIR = "data/train" #　学習データ
TEST_DIR = "data/val"    # テストデータ
BATCH_SIZE = 1           # バッチサイズ
LEARNING_RATE = 1e-5     # 学習率
LAMBDA_IDENTITY = 0.0    # アイデンティティ損失の重み(0.0は無効)
LAMBDA_CYCLE = 10        # サイクル損失の重み
NUM_WORKERS = 4          # データローダーのワーカー数
NUM_EPOCHS = 10          # エポック数
LOAD_MODEL = False       # モデルのロードフラグ
SAVE_MODEL = True        # モデルの保存フラグ
CHECKPOINT_GEN_H = "genh.pth.tar"          # Generator H(Horse to Zebra)
CHECKPOINT_GEN_Z = "genz.pth.tar"          # Generator Z(Zebra to Horse)
CHECKPOINT_CRITIC_H = "critich.pth.tar"    # Critic H(Horse)
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"    # Critic Z(Zebra)
 
# データ拡張の設定
transforms = A.Compose(
    [
        A.Resize(width=256, height=256), # 256x256にリサイズ
        A.HorizontalFlip(p=0.5),         # 50%の確率で水平反転
        A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], max_pixel_value=255),  # [-1,1]に正規化
        ToTensorV2(), # PyTorchテンソルに変換
    ],
    additional_targets={"image0":"image"}, # 2つの画像を同時に変換
)
