# CycleGAN Implementation

PyTorchを使用したCycleGANの実装です。馬とシマウマの画像変換を行います。

## プロジェクト概要

このプロジェクトは、CycleGAN（Cycle-Consistent Adversarial Networks）を用いて、馬とシマウマの画像を相互変換するモデルを学習・実行するためのコードです。

## ディレクトリ構造

```
my-cyclegan/
├── train.py              # メインの学習スクリプト
├── train_exp.py          # 実験用学習スクリプト（詳細なコメント付き）
├── config.py             # 設定ファイル
├── config_exp.py         # 実験用設定ファイル
├── dataset.py            # カスタムデータセットクラス
├── dataset_exp.py        # 実験用データセットクラス
├── generator_model.py    # Generator（生成器）モデル
├── discriminator_model.py # Discriminator（判別器）モデル
├── utils.py              # ユーティリティ関数
├── image_confirm.py      # 画像確認スクリプト
├── image_func.py         # 画像処理関数
├── datasets/             # データセット格納ディレクトリ
│   └── horse2zebra/
│       ├── trainA/       # 馬の学習画像
│       ├── trainB/       # シマウマの学習画像
│       ├── testA/        # 馬のテスト画像
│       └── testB/        # シマウマのテスト画像
├── data/                 # 追加データ格納ディレクトリ
└── save_images/          # 生成画像保存ディレクトリ
```

## 必要な環境

- Python 3.8+
- PyTorch 1.9+
- torchvision
- albumentations
- numpy
- PIL (Pillow)
- matplotlib
- tqdm

## インストール

```bash
pip install torch torchvision torchaudio
pip install albumentations opencv-python
pip install numpy pillow matplotlib tqdm
```

## データセットの準備

1. Horse2Zebraデータセットをダウンロード
2. `datasets/horse2zebra/` ディレクトリに配置
   - `trainA/`: 馬の学習画像
   - `trainB/`: シマウマの学習画像
   - `testA/`: 馬のテスト画像
   - `testB/`: シマウマのテスト画像

## 使用方法

### 1. データの確認

```bash
python image_confirm.py
```

学習データからランダムに選択された馬とシマウマの画像を表示します。

### 2. 学習の実行

```bash
# 基本版
python train.py

# 実験版（詳細なコメント付き）
python train_exp.py
```

### 3. 設定の変更

[`config.py`](config.py) または [`config_exp.py`](config_exp.py) で以下の設定を変更できます：

- `BATCH_SIZE`: バッチサイズ（デフォルト: 1）
- `LEARNING_RATE`: 学習率（デフォルト: 1e-5）
- `NUM_EPOCHS`: エポック数（デフォルト: 10）
- `LAMBDA_CYCLE`: サイクル損失の重み（デフォルト: 10）
- `LAMBDA_IDENTITY`: アイデンティティ損失の重み（デフォルト: 0.0）

## モデル詳細

### Generator
- U-Net + ResNet風アーキテクチャ
- 9個の残差ブロック
- 入力・出力サイズ: 256x256x3

### Discriminator
- PatchGAN構造（70x70パッチ）
- インスタンス正規化使用
- LeakyReLU活性化関数

## 損失関数

1. **敵対的損失（Adversarial Loss）**: 生成画像の品質向上
2. **サイクル一貫性損失（Cycle Consistency Loss）**: A→B→A'でA≈A'を保証
3. **アイデンティティ損失（Identity Loss）**: 同ドメイン入力時の変化抑制

## 学習の監視

- 200ステップごとに生成画像が `save_images/` に保存されます
- プログレスバーで判別器の性能を確認できます
- チェックポイントが自動保存されます

## ファイル説明

| ファイル | 説明 |
|---------|------|
| [`train.py`](train.py) | メインの学習スクリプト |
| [`train_exp.py`](train_exp.py) | コメント付き実験用学習スクリプト |
| [`generator_model.py`](generator_model.py) | Generator（生成器）の定義 |
| [`discriminator_model.py`](discriminator_model.py) | Discriminator（判別器）の定義 |
| [`dataset.py`](dataset.py) | カスタムデータセットクラス |
| [`utils.py`](utils.py) | チェックポイント保存・読み込み関数 |
| [`config.py`](config.py) | 学習設定・ハイパーパラメータ |

## 学習済みモデルの保存・読み込み

```python
# 保存
if config.SAVE_MODEL:
    save_checkpoint(gen_A, opt_gen, filename="gen_A.pth.tar")

# 読み込み
if config.LOAD_MODEL:
    load_checkpoint("gen_A.pth.tar", gen_A, opt_gen, config.LEARNING_RATE)
```

## トラブルシューティング

### よくあるエラー

1. **CUDA out of memory**: バッチサイズを減らす
2. **データセットが見つからない**: データセットパスを確認
3. **依存関係エラー**: 必要なライブラリがインストールされているか確認

### パフォーマンス最適化

- 混合精度学習（AMP）を使用して高速化
- `num_workers`を調整してデータ読み込みを並列化
- GPU使用時は `pin_memory=True` でメモリ転送を高速化

## 参考文献

- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [Horse2Zebra Dataset](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。
