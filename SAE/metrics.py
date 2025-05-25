# metrics.py ---------------------------------------------------------------
import torch, math
import torch.nn.functional as F

@torch.no_grad()
def eval_sae(sae, loader, device):
    sae.eval().to(device)

    mse_total      = 0.0
    cos_total      = 0.0
    sparsity_total = 0.0
    n_tokens       = 0

    # allocate **boolean** mask – one slot per hidden unit
    H = sae.encoder.out_features
    alive_mask = torch.zeros(H, dtype=torch.bool, device=device)

    for x in loader:                           # x  ∈  ℝ[B,1152]
        x   = x.to(device)
        x̂, z, _ = sae(x)                      # forward pass

        mse_total      += F.mse_loss(x̂, x, reduction="sum").item()
        cos_total      += F.cosine_similarity(x̂, x, dim=1).sum().item()
        sparsity_total += z.abs().sum().item()
        n_tokens       += x.size(0)

        # update the “unit ever fired” bitmap
        alive_mask |= (z.abs() > 1e-5).any(dim=0)   # both tensors are bool now ✔

    return {
        "mse"        : mse_total / n_tokens,
        "cos_sim"    : cos_total / n_tokens,
        "sparsity"   : sparsity_total / n_tokens,
        "dead_ratio" : 1.0 - alive_mask.float().mean().item()
    }
