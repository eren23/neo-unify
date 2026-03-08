"""Run all Neo-Unify PyTorch experiments sequentially."""

import time


def main():
    print("=" * 60)
    print("Neo-Unify PyTorch: Running All Experiments")
    print("=" * 60)
    t0 = time.time()

    # 1. Train VQ-VAE
    print("\n" + "=" * 60)
    print("Step 1/8: Training VQ-VAE")
    print("=" * 60)
    from neo_unify_torch.vqvae.train import train as train_vqvae
    train_vqvae()

    # 2. Train Phase C
    print("\n" + "=" * 60)
    print("Step 2/8: Training Phase C (Neo-Unify MoT)")
    print("=" * 60)
    from neo_unify_torch.phase_c.train import train as train_phase_c
    train_phase_c()

    # 3. Generate Phase C
    print("\n" + "=" * 60)
    print("Step 3/8: Generating Phase C")
    print("=" * 60)
    from neo_unify_torch.phase_c.generate import generate as gen_phase_c
    gen_phase_c()

    # 4. Train Exp CFG
    print("\n" + "=" * 60)
    print("Step 4/8: Training Exp CFG")
    print("=" * 60)
    from neo_unify_torch.exp_cfg.train import train as train_exp_cfg
    train_exp_cfg()

    # 5. Generate Exp CFG
    print("\n" + "=" * 60)
    print("Step 5/8: Generating Exp CFG")
    print("=" * 60)
    from neo_unify_torch.exp_cfg.generate import generate as gen_exp_cfg
    gen_exp_cfg()

    # 6. Train Exp Latent
    print("\n" + "=" * 60)
    print("Step 6/8: Training Exp Latent")
    print("=" * 60)
    from neo_unify_torch.exp_latent.train import train as train_exp_latent
    train_exp_latent()

    # 7. Generate Exp Latent
    print("\n" + "=" * 60)
    print("Step 7/8: Generating Exp Latent")
    print("=" * 60)
    from neo_unify_torch.exp_latent.generate import generate as gen_exp_latent
    gen_exp_latent()

    # 8. Compare
    print("\n" + "=" * 60)
    print("Step 8/8: Comparing All Methods")
    print("=" * 60)
    from neo_unify_torch.compare import compare
    compare()

    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"All experiments complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
