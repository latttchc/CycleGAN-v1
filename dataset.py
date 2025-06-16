import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class HorseZebraDataset(Dataset):
    """
    CycleGAN用のカスタムデータセットクラス
    馬とシマウマの画像を同時に読み込み、ペアで返す
    
    Note:
        - 2つのドメイン（馬/シマウマ）のデータを扱う
        - データ数が異なる場合は循環的にサンプリング
        - Albumentationsによる同期データ拡張をサポート
    """
    
    def __init__(self, root_horse, root_zebra, transform=None):
        """
        データセットの初期化
        
        Args:
            root_horse: 馬画像のディレクトリパス
            root_zebra: シマウマ画像のディレクトリパス  
            transform: データ拡張用の変換関数（Albumentations）
        """
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform

        # 各ディレクトリから画像ファイル名を取得
        self.zebra_images = os.listdir(root_zebra)
        self.horse_images = os.listdir(root_horse)
        
        # データセット長は多い方に合わせる（少ない方は循環）
        self.length_dataset = max(len(self.zebra_images), len(self.horse_images))
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)

    def __len__(self):
        """
        データセットのサイズを返す
        
        Returns:
            int: データセットの総サンプル数
        """
        return self.length_dataset

    def __getitem__(self, index):
        """
        指定されたインデックスのデータを取得
        
        Args:
            index: データのインデックス
            
        Returns:
            tuple: (シマウマ画像テンソル, 馬画像テンソル)
        """
        # インデックスが範囲外の場合は循環的にサンプリング
        zebra_img = self.zebra_images[index % self.zebra_len]
        horse_img = self.horse_images[index % self.horse_len]

        # ファイルパスの構築
        zebra_path = os.path.join(self.root_zebra, zebra_img)
        horse_path = os.path.join(self.root_horse, horse_img)

        # 画像を読み込み、RGBに変換してNumPy配列化
        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        # データ拡張の適用（両画像に同じ変換を適用）
        if self.transform:
            augmentations = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augmentations["image"]      # メイン画像
            horse_img = augmentations["image0"]     # 追加画像
            
        return zebra_img, horse_img