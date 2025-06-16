import torch
import torch.nn as nn

class Block(nn.Module):
    """
    Discriminatorの基本ブロック
    Conv2d + InstanceNorm2d + LeakyReLUの組み合わせ
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            # 4x4カーネル、reflectパディングを使用
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),  # バッチ正規化の代わりにインスタンス正規化
            nn.LeakyReLU(0.2),  # 負の傾き0.2のLeakyReLU
        )
    
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator
    画像が本物か偽物かを70x70のパッチレベルで判断する
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        # 最初の層（InstanceNormなし）
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        # 中間層の構築
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            # 最後の層以外はstride=2、最後の層はstride=1
            layers.append(Block(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        
        # 最終出力層（1チャンネル出力）
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向き伝播
        Args:
            x: 入力画像 (batch_size, channels, height, width)
        Returns:
            出力確率マップ (batch_size, 1, patch_height, patch_width)
        """
        x = self.initial(x)  # 最初の畳み込み
        return torch.sigmoid(self.model(x))  # シグモイド関数で0-1の確率に変換
    
def test():
    """
    Discriminatorモデルのテスト関数
    256x256の入力に対して30x30のパッチ出力を確認
    """
    x = torch.randn((5, 3, 256, 256))  # バッチサイズ5、3チャンネル、256x256画像
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(f"出力形状: {preds.shape}")  # (5, 1, 30, 30)になるはず

if __name__ == "__main__":
    test()


