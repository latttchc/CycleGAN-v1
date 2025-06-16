import torch
from dataset import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def train_fn(
    disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    """
    CycleGANの1エポック分の学習を実行
    
    Args:
        disc_H: 馬の判別器（Horse Discriminator）
        disc_Z: シマウマの判別器（Zebra Discriminator）
        gen_Z: 馬→シマウマ生成器（Horse to Zebra Generator）
        gen_H: シマウマ→馬生成器（Zebra to Horse Generator）
        loader: データローダー
        opt_disc: 判別器のオプティマイザー
        opt_gen: 生成器のオプティマイザー
        l1: L1損失関数（サイクル一貫性・アイデンティティ損失用）
        mse: MSE損失関数（敵対的損失用）
        d_scaler: 判別器用混合精度スケーラー
        g_scaler: 生成器用混合精度スケーラー
    """
    # 判別器の性能モニタリング用変数
    H_reals = 0  # 本物の馬に対する判別器スコアの累計
    H_fakes = 0  # 偽の馬に対する判別器スコアの累計
    loop = tqdm(loader, leave=True)  # プログレスバー

    for idx, (zebra, horse) in enumerate(loop):
        # データをGPUに移動
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # ========== 判別器の学習 ==========
        # 混合精度学習で計算の高速化
        with torch.cuda.amp.autocast():
            # 1. 馬判別器の学習
            fake_horse = gen_H(zebra)  # シマウマから偽の馬を生成
            D_H_real = disc_H(horse)   # 本物の馬を判別
            D_H_fake = disc_H(fake_horse.detach())  # 偽の馬を判別（勾配伝播を停止）
            
            # 判別器の性能をモニタリング
            H_reals += D_H_real.mean().item()  # 本物に対するスコア（1に近いほど良い）
            H_fakes += D_H_fake.mean().item()  # 偽物に対するスコア（0に近いほど良い）
            
            # 損失計算：本物は1、偽物は0として学習
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            # 2. シマウマ判別器の学習
            fake_zebra = gen_Z(horse)  # 馬から偽のシマウマを生成
            D_Z_real = disc_Z(zebra)   # 本物のシマウマを判別
            D_Z_fake = disc_Z(fake_zebra.detach())  # 偽のシマウマを判別
            
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # 両判別器の損失を統合
            D_loss = (D_H_loss + D_Z_loss) / 2

        # 判別器パラメータの更新
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # ========== 生成器の学習 ==========
        with torch.cuda.amp.autocast():
            # 1. 敵対的損失（Adversarial Loss）
            # 生成器は判別器を騙そうとする（偽物を本物として判定させる）
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))  # 馬生成器の敵対的損失
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))  # シマウマ生成器の敵対的損失

            # 2. サイクル一貫性損失（Cycle Consistency Loss）
            # A → B → A' で元の画像A ≈ A'になることを保証
            cycle_zebra = gen_Z(fake_horse)  # 偽馬 → シマウマ（元のシマウマに戻るはず）
            cycle_horse = gen_H(fake_zebra)  # 偽シマウマ → 馬（元の馬に戻るはず）
            cycle_zebra_loss = l1(zebra, cycle_zebra)  # 元のシマウマとの差
            cycle_horse_loss = l1(horse, cycle_horse)  # 元の馬との差

            # 3. アイデンティティ損失（Identity Loss）
            # 同じドメインの画像を入力した時は変化しないことを保証
            # G_H(horse) ≈ horse, G_Z(zebra) ≈ zebra
            identity_zebra = gen_Z(zebra)  # シマウマ→シマウマ（変化なし）
            identity_horse = gen_H(horse)  # 馬→馬（変化なし）
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # 総合的な生成器損失
            G_loss = (
                loss_G_Z +                                          # シマウマ生成器の敵対的損失
                loss_G_H +                                          # 馬生成器の敵対的損失
                cycle_zebra_loss * config.LAMBDA_CYCLE +            # シマウマのサイクル損失
                cycle_horse_loss * config.LAMBDA_CYCLE +            # 馬のサイクル損失
                identity_horse_loss * config.LAMBDA_IDENTITY +      # 馬のアイデンティティ損失
                identity_zebra_loss * config.LAMBDA_IDENTITY        # シマウマのアイデンティティ損失
            )

        # 生成器パラメータの更新
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # 200ステップごとに生成画像を保存（学習進捗の確認用）
        if idx % 200 == 0:
            # [-1,1] → [0,1] に正規化してから保存
            save_image(fake_horse * 0.5 + 0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_zebra * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")

        # プログレスバーに判別器の性能を表示
        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


def main():
    """
    メイン関数：モデルの初期化、データローダーの作成、学習ループの実行
    """
    # ========== モデルの初期化 ==========
    # CycleGANには4つのネットワークが必要：2つの生成器 + 2つの判別器
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)  # 馬判別器
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)  # シマウマ判別器
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)  # 馬→シマウマ生成器
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)  # シマウマ→馬生成器
    
    # ========== オプティマイザーの設定 ==========
    # 判別器用：両方の判別器のパラメータを統合
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),  # GANで推奨される設定（momentum減衰を遅く）
    )
    
    # 生成器用：両方の生成器のパラメータを統合
    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    # ========== 損失関数の定義 ==========
    L1 = nn.L1Loss()    # L1距離（サイクル一貫性・アイデンティティ損失用）
    mse = nn.MSELoss()  # 平均二乗誤差（敵対的損失用）

    # ========== チェックポイントの読み込み ==========
    # 学習を再開する場合
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE)

    # ========== データセットとデータローダーの作成 ==========
    # 学習用データセット
    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR + "/horses",
        root_zebra=config.TRAIN_DIR + "/zebras",
        transform=config.transforms,
    )
    
    # データローダーの作成
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,      # 学習用バッチサイズ
        shuffle=True,                      # データをシャッフル
        num_workers=config.NUM_WORKERS,    # 並列データ読み込み
        pin_memory=True,                   # GPU転送の高速化
    )
    
    # ========== 混合精度学習の設定 ==========
    # 学習の高速化とメモリ使用量削減
    g_scaler = torch.cuda.amp.GradScaler()  # 生成器用
    d_scaler = torch.cuda.amp.GradScaler()  # 判別器用

    # ========== 学習ループ ==========
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
        train_fn(
            disc_H, disc_Z, gen_Z, gen_H, loader,
            opt_disc, opt_gen, L1, mse, d_scaler, g_scaler,
        )

        # エポック終了時にモデルを保存
        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)


if __name__ == "__main__":
    main()