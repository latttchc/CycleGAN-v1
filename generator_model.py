import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    畳み込みブロックの基本構成要素
    Conv2d/ConvTranspose2d + InstanceNorm2d + ReLUの組み合わせ
    
    Args:
        in_channels: 入力チャンネル数
        out_channels: 出力チャンネル数
        down: True=ダウンサンプリング(Conv2d), False=アップサンプリング(ConvTranspose2d)
        use_act: True=ReLU活性化関数を使用, False=Identity(活性化なし)
        **kwargs: カーネルサイズ、ストライド、パディングなどの追加パラメータ
    """
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            # ダウンサンプリング：通常の畳み込み（reflect padding使用）
            # アップサンプリング：転置畳み込み
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            
            # バッチ正規化の代わりにインスタンス正規化（GANでより安定）
            nn.InstanceNorm2d(out_channels),
            
            # 活性化関数（最後の層では使わない場合がある）
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)
    
class ResidualBlock(nn.Module):
    """
    残差ブロック（ResNet風）
    入力をスキップ接続で出力に加算することで勾配消失を防ぐ
    
    Args:
        channels: 入力・出力チャンネル数（残差ブロックでは同じ）
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            # 1つ目の畳み込み（ReLUあり）
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            # 2つ目の畳み込み（ReLUなし、残差接続前）
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # 残差接続：入力 + 変換結果
        return x + self.block(x)

class Generator(nn.Module):
    """
    CycleGAN用Generator（U-Net + ResNet風アーキテクチャ）
    
    構造：
    入力 → 初期畳み込み → ダウンサンプリング → 残差ブロック群 → アップサンプリング → 出力
    
    Args:
        img_channels: 入力画像のチャンネル数（通常3=RGB）
        num_features: 基本特徴量数（デフォルト64）
        num_residuals: 残差ブロックの数（デフォルト9）
    """
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        
        # 1. 初期畳み込み層（7x7カーネル、reflect padding）
        # RGB → 64チャンネルの特徴マップに変換
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )
        
        # 2. ダウンサンプリング層（エンコーダー部分）
        # 画像サイズを縮小しながら特徴を抽出
        # 256x256 → 128x128 → 64x64
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),      # 64→128ch
                ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),    # 128→256ch
            ]
        )

        # 3. 残差ブロック群（ボトルネック部分）
        # 高次元特徴空間で画像変換を学習
        # 9個の残差ブロックで複雑な変換を可能にする
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]  # 256chで9回
        )
        
        # 4. アップサンプリング層（デコーダー部分）
        # 特徴マップから元の画像サイズに復元
        # 64x64 → 128x128 → 256x256
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256→128ch
                ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128→64ch
            ]
        )

        # 5. 最終出力層（7x7カーネル、reflect padding）
        # 64チャンネル → RGBの3チャンネルに変換
        self.last = nn.Conv2d(num_features*1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        """
        前向き伝播
        
        Args:
            x: 入力画像 (batch_size, 3, 256, 256)
            
        Returns:
            変換された画像 (batch_size, 3, 256, 256), 値域[-1, 1]
        """
        # 1. 初期畳み込み
        x = self.initial(x)
        
        # 2. ダウンサンプリング（エンコード）
        for layer in self.down_blocks:
            x = layer(x)
        
        # 3. 残差ブロック群で特徴変換
        x = self.residual_blocks(x)
        
        # 4. アップサンプリング（デコード）
        for layer in self.up_blocks:
            x = layer(x)
        
        # 5. 最終出力（tanh関数で[-1,1]に正規化）
        return torch.tanh(self.last(x))
    
def test():
    """
    Generatorモデルのテスト関数
    入力と出力のサイズが一致することを確認
    """
    img_channels = 3        # RGB画像
    img_size = 256         # 256x256サイズ
    x = torch.randn((2, img_channels, img_size, img_size))  # バッチサイズ2のダミー入力
    gen = Generator(img_channels, 9)
    print(f"入力形状: {x.shape}")
    print(f"出力形状: {gen(x).shape}")  # (2, 3, 256, 256)になるはず

if __name__ == "__main__":
    test()
