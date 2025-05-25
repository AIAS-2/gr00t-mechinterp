# --------------------------------------------------------------------------- #
#  train_sae_ddp.py – DDP training *with one final metrics pass*              #
# --------------------------------------------------------------------------- #
import os, json, torch, matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path

from sparse_autoencoder import SparseAutoencoder, TokenActivationDataset
from metrics            import eval_sae              # <- we still reuse this

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Distributed helpers
# ─────────────────────────────────────────────────────────────────────────────
def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # ─── experiment hyper-params ──────────────────────────────────────────
    layer_idx  = 11                       # ViT layer to probe
    img_dir    = Path("openimages_subset/images_train")
    paths      = [str(p) for p in img_dir.iterdir()]
    n_epochs   = 5
    batch_size = 128                      # *per* GPU
    lr         = 1e-3

    # ─── frozen ViT / tokenizer (same helper you already have) ───────────
    from train_sae import load_models
    vit, txt, tok = load_models(device)

    # ─── dataset / dataloader ────────────────────────────────────────────
    ds       = TokenActivationDataset(vit, paths, layer_idx, device)
    sampler  = DistributedSampler(ds,
                                  num_replicas=world_size,
                                  rank=rank,
                                  shuffle=True)
    dl       = DataLoader(ds,
                          batch_size=batch_size,
                          sampler=sampler,
                          num_workers=0,
                          pin_memory=False,
                          drop_last=True)

    # ─── sparse auto-encoder + DDP wrapper ───────────────────────────────
    sae      = SparseAutoencoder(input_dim=1152,
                                 hidden_dim=512,
                                 sparsity_weight=1e-3).to(device)
    sae_ddp  = DDP(sae, device_ids=[local_rank])

    opt = torch.optim.Adam(sae_ddp.parameters(), lr=lr)
    train_loss_history = []               # we only track training loss now

    # ──────────────────────────────────────────────────────────────────
    # 2.  TRAINING LOOP (no per-epoch validation anymore)
    # ──────────────────────────────────────────────────────────────────
    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        sae_ddp.train()

        epoch_loss = 0.0
        for batch in dl:
            batch = batch.to(device, non_blocking=True)
            _, _, loss = sae_ddp(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        # rank-0 prints the average loss for the epoch
        if rank == 0:
            avg_loss = epoch_loss / len(dl)
            train_loss_history.append(avg_loss)
            print(f"[{epoch+1:02d}/{n_epochs}]  train-loss = {avg_loss:.5f}")

        # keep all ranks in sync
        dist.barrier()

    # ──────────────────────────────────────────────────────────────────
    # 3.  AFTER TRAINING:  evaluate once and save everything (rank-0)
    # ──────────────────────────────────────────────────────────────────
    if rank == 0:
        # 3-a) checkpoint
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(sae.state_dict(), ckpt_dir / f"512_sae_layer{layer_idx+1}.pt")

        # 3-b) quick validation on 2 % of the tokens
        val_subset = torch.utils.data.Subset(ds, range(len(ds) // 50))
        val_loader = DataLoader(val_subset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0)
        final_stats = eval_sae(sae, val_loader, device)

        # 3-c) plots ------------------------------------------------------
        epochs = range(1, n_epochs + 1)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        # left: training-loss curve
        ax[0].plot(epochs, train_loss_history, label="train loss")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("MSE + λ|z|")
        ax[0].legend()
        ax[0].grid(True)

        # right: final metrics as a bar chart
        ax[1].bar(range(len(final_stats)), final_stats.values())
        ax[1].set_xticks(range(len(final_stats)), final_stats.keys(), rotation=45)
        ax[1].set_ylabel("value")
        ax[1].set_title("final validation metrics")
        ax[1].grid(True)

        fig.tight_layout()
        fig.savefig(ckpt_dir / "512_training_curves.png")

        # 3-d) raw numbers
        with open(ckpt_dir / "512_metrics.json", "w") as fp:
            json.dump(
                {"train_loss": train_loss_history, "final_metrics": final_stats},
                fp,
                indent=2,
            )

    # ──────────────────────────────────────────────────────────────────
    cleanup_ddp()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
