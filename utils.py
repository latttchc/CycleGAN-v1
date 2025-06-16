import random, torch, os, numpy as np
import config

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """
    モデルとオプティマイザーの状態をファイルに保存する
    
    Args:
        model: 保存するPyTorchモデル
        optimizer: 保存するオプティマイザー
        filename: 保存先のファイル名（デフォルト: "my_checkpoint.pth.tar"）
    """
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),  # モデルのパラメータ
        "optimizer": optimizer.state_dict(),  # オプティマイザーの状態
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """
    保存されたチェックポイントからモデルとオプティマイザーの状態を復元する
    
    Args:
        checkpoint_file: 読み込むチェックポイントファイルのパス
        model: 状態を復元するPyTorchモデル
        optimizer: 状態を復元するオプティマイザー
        lr: 設定する学習率
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])  # モデルパラメータの復元
    optimizer.load_state_dict(checkpoint["optimizer"])  # オプティマイザー状態の復元

    # 学習率を指定された値に更新
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def seed_everything(seed=42):
    """
    再現可能な結果を得るために、すべての乱数生成器のシードを固定する
    
    Args:
        seed: 設定する乱数シード値（デフォルト: 42）
    
    Note:
        機械学習の実験で結果を再現可能にするために使用
        トレーニングの開始時に一度だけ呼び出す
    """
    # Python標準の乱数生成器
    os.environ["PYTHONHASHSEED"] = str(seed)  # Pythonハッシュシードを固定
    random.seed(seed)  # Python randomモジュールのシード
    
    # NumPyの乱数生成器
    np.random.seed(seed)
    
    # PyTorchの乱数生成器
    torch.manual_seed(seed)  # CPU用
    torch.cuda.manual_seed(seed)  # 単一GPU用
    torch.cuda.manual_seed_all(seed)  # 複数GPU用
    
    # CuDNNの動作を決定的にする（速度は犠牲になる）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
