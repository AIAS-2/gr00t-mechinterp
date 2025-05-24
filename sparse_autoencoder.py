# --------------------------- sparse_autoencoder.py ---------------------------
import os, json, random
from pathlib import Path
from typing import Sequence, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms as T

# ---------- 1.  An SAE module with an L1 sparsity penalty --------------------
class SparseAutoencoder(nn.Module):
    """
    One-hidden-layer sparse auto-encoder.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 sparsity_weight: float = 1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        z = F.relu(self.encoder(x))                   # [B, hidden_dim]
        x_hat = self.decoder(z)                       # [B, input_dim]
        recon_loss = F.mse_loss(x_hat, x)
        sparsity_loss = z.abs().mean()
        loss = recon_loss + self.sparsity_weight * sparsity_loss
        return x_hat, z, loss

# ---------- 2.  Dataset that streams *token vectors* from a ViT -------------
class TokenActivationDataset(Dataset):
    """
    Each item = one patch-token vector taken from the chosen ViT layer.

    Parameters
    ----------
    vit_model     : the frozen SigLIP vision encoder (already on the right device)
    image_paths   : list/tuple of str – paths to image files
    layer_idx     : which hidden-states layer to read (0-based)
    device        : torch.device
    return_meta   : if **True**, __getitem__ returns
                    (vec, img_path, patch_idx) so that callers can
                    trace back the token; if False (default) it returns only
                    the vector – that is what the SAE trainer expects.
    """
    def __init__(self,
                 vit_model,
                 image_paths: Sequence[str],
                 layer_idx: int,
                 device: torch.device,
                 return_meta: bool = False):

        self.vit        = vit_model.eval().to(device)
        self.paths      = list(image_paths)
        self.layer_idx  = layer_idx
        self.device     = device
        self.return_meta= return_meta

        self.tfm = T.Compose([
            T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # How many tokens per image?  (needed for __len__)
        with torch.no_grad():
            dummy = (
                self.tfm(Image.new("RGB", (448, 448)))
                .unsqueeze(0)
                .to(device, dtype=torch.float16)
            )
            n_tokens = self.vit(pixel_values=dummy,
                                output_hidden_states=True
                               ).hidden_states[layer_idx].size(1)
        self.tokens_per_img = n_tokens
        print(f"[SAE] layer {layer_idx}: each image ⇒ {n_tokens} tokens")

    # --------------------------- Dataset API ---------------------------
    def __len__(self):
        return len(self.paths) * self.tokens_per_img

    def __getitem__(self, idx):
        img_idx   = idx // self.tokens_per_img       # which picture
        token_idx = idx %  self.tokens_per_img       # which patch token

        path = self.paths[img_idx]
        img  = Image.open(path).convert("RGB")

        x = (
            self.tfm(img)
            .unsqueeze(0)
            .to(self.device, dtype=torch.float16)
        )

        with torch.no_grad():
            h = self.vit(pixel_values=x,
                         output_hidden_states=True).hidden_states[self.layer_idx]

        vec = h[0, token_idx]                # shape = (hidden_dim,)

        if self.return_meta:
            # for inspection: return the vector **plus** provenance info
            return vec.float(), path, token_idx
        return vec.float()


# ---------- 3.  Simple training routine -------------------------------------
def train_sae(sae: SparseAutoencoder,
              train_loader: DataLoader,
              n_epochs: int = 5,
              lr: float = 1e-3,
              device: torch.device = "cuda"):
    sae.to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch in train_loader:           # batch = [B, 1152]
            batch = batch.to(device)
            _, _, loss = sae(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"[SAE] epoch {epoch+1}/{n_epochs} – loss {epoch_loss/len(train_loader):.5f}")

# ---------- 4.  Utility: show top-activating patches for each SAE unit -------
def top_k_patches(sae: SparseAutoencoder,
                  loader: DataLoader,
                  k: int = 20,
                  device: torch.device = "cuda"):
    """
    Returns a dict {unit_id: [tensor(…, hidden_dim)] * k}
    containing the patch vectors with the highest activation value.
    """
    sae.eval().to(device)
    top = {}                                  # unit → list of (act, vec)
    with torch.no_grad():
        for vec in loader:
            vec = vec.to(device)
            z = F.relu(sae.encoder(vec))      # [B, hidden_dim]
            for unit in range(z.size(1)):
                acts = z[:, unit]             # [B]
                for a, v in zip(acts, vec):
                    if unit not in top:
                        top[unit] = []
                    top[unit].append((a.item(), v.cpu()))
                    top[unit] = sorted(top[unit], key=lambda t: -t[0])[:k]
    # strip scores
    return {u: [v for _, v in lst] for u, lst in top.items()}
