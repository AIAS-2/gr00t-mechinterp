# ──────────────────────────  inspect_sae.py  ──────────────────────────
"""
Analyse a trained Sparse Auto-encoder (SAE) on fresh images.

Example
-------
python inspect_sae.py \
        --ckpt        checkpoints/sae_layer12.pt \
        --image_dir   new_images/ \
        --layer       11 \
        --top_k       30 \
        --out_dir     sae_results/ \
        --auto_label          # (optional) add semantic labels with CLIP
"""

import os, math, argparse, json, collections
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np

from PIL import Image

# ===== your own modules  =====
from sparse_autoencoder import (
        SparseAutoencoder,
        TokenActivationDataset
)
from train_sae import load_models          # ← already defined in your train script

# ===== optional CLIP for auto-labels =====
try:
    from transformers import AutoTokenizer, AutoModel
    CLIP_NAME = "openai/clip-vit-large-patch14"
except ImportError:
    CLIP_NAME = None
    
import os

# Set up Hugging Face cache paths
scratch_dir = os.path.expanduser("~/scratch/huggingface")
os.makedirs(scratch_dir, exist_ok=True)
os.environ["HF_HOME"] = scratch_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(scratch_dir, "transformers")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(scratch_dir, "hub")


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
@torch.no_grad()
def collect_top_patches(sae, vit, paths, layer_idx,
                        device="cuda", top_k=30, batch_size=32):
    """
    Much faster version:  one ViT pass per image.

    Returns dict: unit_id → list[(act_score, img_path, patch_idx)] len = top_k
    """
    
    import torchvision.transforms as T
    
    sae  = sae.eval().to(device)
    vit  = vit.eval().to(device)

    # simple preprocessing pipeline (same as your dataset)
    tfm = T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # reservoir of top-k activations per unit
    heap = collections.defaultdict(list)

    # batched inference – 32 images at a time keeps GPU busy
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        imgs = torch.stack([
            tfm(Image.open(p).convert("RGB")) for p in batch_paths
        ]).to(device, dtype=torch.float16)                     # [B,3,448,448]

        # 1) ViT once per image
        h = vit(pixel_values=imgs, output_hidden_states=True
               ).hidden_states[layer_idx]                      # [B, 1024, 1152]

        # 2) flatten patches to shape [B*1024, 1152] and encode with SAE
        B, T, D = h.shape
        vecs = h.reshape(B*T, D).float()                       # fp32 for SAE
        acts = F.relu(sae.encoder(vecs))                      # [B*T, H]

        # 3) walk over activations
        for b, pth in enumerate(batch_paths):
            base = b*T
            for tkn in range(T):
                row = base + tkn
                for unit, a in enumerate(acts[row]):
                    a = a.item()
                    lst = heap[unit]
                    tpl = (a, pth, tkn)
                    if len(lst) < top_k:
                        lst.append(tpl)
                        lst.sort(key=lambda x: -x[0])
                    elif a > lst[-1][0]:
                        lst[-1] = tpl
                        lst.sort(key=lambda x: -x[0])

        if (i + batch_size) % 100 == 0 or i + batch_size >= len(paths):
            print(f"... analysed {min(i+batch_size,len(paths))} / {len(paths)} images", flush=True)

    return heap


def patch_bbox(patch_idx, patch_size=14, img_size=448):
    grid = img_size // patch_size
    r, c = divmod(patch_idx, grid)
    return (c*patch_size, r*patch_size,
            (c+1)*patch_size, (r+1)*patch_size)


def montage(unit_top, out_path, patch_size=14, grid_w=None, scale=6):
    """
    Save a visual grid for the top-k patches of one unit.

    • grid_w=None → automatically pick a square-ish grid.
    • scale > 1   → up-sample each patch for readability.
    """
    if grid_w is None:
        grid_w = math.ceil(math.sqrt(len(unit_top)))

    s = patch_size * scale                      # visual size of each tile
    grid_h = math.ceil(len(unit_top) / grid_w)
    canvas = Image.new("RGB", (grid_w * s, grid_h * s), color=(40, 40, 40))

    for idx, (_, img_path, pidx) in enumerate(unit_top):
        img = Image.open(img_path).convert("RGB")
        tile = img.crop(patch_bbox(pidx, patch_size))
        tile = tile.resize((s, s), Image.NEAREST)   # or BICUBIC for smooth
        x = (idx % grid_w) * s
        y = (idx // grid_w) * s
        canvas.paste(tile, (x, y))

    canvas.save(out_path)
    
    
# ---------------------------------------------------------------------
# quantitative metrics
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate_metrics(sae, vit, paths, layer_idx,
                     device="cuda", batch_size=32):
    """
    Returns:
        mse_mean   : average reconstruction MSE per patch vector
        spars_mean : average sparsity ratio  (#zeros / hidden_dim)
    """
    sae  = sae.eval().to(device)
    vit  = vit.eval().to(device)

    import torchvision.transforms as T
    tfm = T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    mse_tot, spars_tot, n_vec = 0.0, 0.0, 0

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        imgs = torch.stack([
            tfm(Image.open(p).convert("RGB")) for p in batch_paths
        ]).to(device, dtype=torch.float16)                     # [B,3,448,448]

        h = vit(pixel_values=imgs, output_hidden_states=True
               ).hidden_states[layer_idx]                      # [B,1024,1152]
        vecs = h.reshape(-1, h.size(-1)).float()               # [B*1024,1152]

        x_hat, z, _ = sae(vecs)                                # recon & latent
        mse_tot   += F.mse_loss(x_hat, vecs, reduction='sum').item()
        spars_tot += (z == 0).sum().item()
        n_vec     += vecs.size(0)

    mse_mean   = mse_tot   / n_vec
    spars_mean = spars_tot / (n_vec * z.size(1))               # ratio in [0,1]

    return mse_mean, spars_mean



# ---------------------------------------------------------------------
# optional CLIP-based semantic label
# ---------------------------------------------------------------------
from transformers import CLIPModel, CLIPTokenizer

_clip, _tok = None, None                      # module-level cache

@torch.no_grad()
def label_unit_clip(unit_top, device):
    global _clip, _tok

    if CLIP_NAME is None:
        return "<clip unavailable>"

    # ---------- load once --------------------
    if _clip is None:
        _tok  = CLIPTokenizer.from_pretrained(CLIP_NAME)
        _clip = CLIPModel.from_pretrained(CLIP_NAME).eval().to(device)

    # ---------- candidate words --------------
    vocab = [
    # --- people & body parts ----------------------------------------------------
    "person","face","eye","ear","nose","mouth","hand","finger","arm","leg","foot",
    "hair","smile","skin","beard","child","woman","man",
    # --- animals ----------------------------------------------------------------
    "dog","cat","horse","cow","sheep","goat","pig","rabbit","bear","deer","elephant",
    "giraffe","lion","tiger","monkey","zebra","panda","bird","duck","chicken",
    "eagle","owl","fish","shark","whale","dolphin","butterfly","bee","insect",
    # --- plants & nature --------------------------------------------------------
    "tree","grass","leaf","flower","rose","sunflower","mushroom","cactus",
    "forest","mountain","hill","river","lake","waterfall","beach","sea","ocean",
    "sky","cloud","sun","moon","star","snow","ice","sand","rock",
    # --- food & drink -----------------------------------------------------------
    "food","fruit","apple","banana","orange","strawberry","grape","lemon","peach",
    "vegetable","carrot","potato","tomato","broccoli","corn","salad",
    "bread","cake","cookie","pizza","burger","sandwich","noodle","rice",
    "coffee","tea","juice","milk","wine","beer","water","bottle","glass","cup",
    # --- vehicles & related -----------------------------------------------------
    "vehicle","car","truck","bus","van","taxi","train","subway","tram","motorcycle",
    "bicycle","scooter","skateboard","boat","ship","airplane","helicopter","rocket",
    "wheel","tire","road","track","bridge","traffic","street","highway","rail",
    # --- buildings & rooms ------------------------------------------------------
    "building","house","cabin","skyscraper","tower","castle","temple","church",
    "mosque","stadium","bridge","lighthouse","window","door","roof","wall","floor",
    "room","kitchen","bathroom","bedroom","office","classroom","corridor",
    "furniture","chair","sofa","table","desk","bed","cabinet","shelf","lamp",
    # --- tools & objects --------------------------------------------------------
    "tool","hammer","screwdriver","wrench","saw","scissors","knife","fork","spoon",
    "pen","pencil","brush","book","paper","notebook","keyboard","mouse","phone",
    "camera","clock","watch","tv","screen","computer","laptop","tablet","remote",
    "microphone","speaker","headphone","bottle","can","jar","box","bag","backpack",
    "wallet","key","ring","umbrella","helmet","ball","bat","racket","guitar",
    # --- clothing & fabric ------------------------------------------------------
    "clothes","shirt","tshirt","jacket","coat","sweater","dress","skirt","pants",
    "jeans","shorts","sock","shoe","boot","hat","cap","glove","scarf","belt",
    # --- colours & materials ----------------------------------------------------
    "red","green","blue","yellow","orange","purple","pink","brown","black","white",
    "gray","gold","silver","metal","steel","iron","wood","plastic","glass","paper",
    "stone","brick","concrete","fabric","leather","rubber","ceramic",
    # --- patterns / symbols / misc ---------------------------------------------
    "text","number","letter","logo","flag","sign","symbol","map","graph","chart",
    "money","coin","banknote","ticket","passport","credit","barcode","qr",
    "fire","smoke","light","shadow","reflection","numbers"
]

    prompts = [f"a photo of a {w}" for w in vocab]

    # ---------- image embedding --------------
    img_embeds = []
    for _, img_path, pidx in unit_top:
        img = Image.open(img_path).convert("RGB")
        img = img.crop(patch_bbox(pidx)).resize((224, 224), Image.BICUBIC)
        img_t = (
            torch.tensor(np.asarray(img))
            .permute(2, 0, 1).float() / 255
        ).unsqueeze(0).to(device)
        img_embeds.append(_clip.get_image_features(pixel_values=img_t))

    im = torch.stack(img_embeds).mean(0, keepdim=True)      # [1, dim]

    # ---------- text embedding ---------------
    txt = _tok(prompts, padding=True, return_tensors="pt").to(device)
    te  = _clip.get_text_features(**txt)                    # [V, dim]

    # ---------- similarity + argmax ----------
    sims = F.cosine_similarity(im, te, dim=-1)              # [V]
    assert sims.numel() == len(vocab), f"{sims.shape=} vs {len(vocab)=}"
    idx  = int(sims.argmax())
    idx  = min(idx, len(vocab) - 1)                         # final guard

    return vocab[idx]




# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1a) models --------------------------------------------------------
    vit, _, _ = load_models(device)                     # frozen ViT
    sae = SparseAutoencoder(input_dim=1152,
                            hidden_dim=cfg.hidden,
                            sparsity_weight=cfg.l1)
    sae.load_state_dict(torch.load(cfg.ckpt, map_location=device))
    print(f"[✓] SAE loaded from {cfg.ckpt}")
    

    # 1b) quantitative metrics ----------------------------------------
    val_paths = [str(p) for p in Path(cfg.image_dir).glob("*")
                 if p.suffix.lower() in [".jpg", ".png"]]

    mse, spars = evaluate_metrics(sae, vit, val_paths,
                                  layer_idx=cfg.layer, device=device)
    print(f"[metrics]  reconstruction MSE = {mse:.6f}   "
          f"sparsity = {spars*100:.2f}% zeros")
    
    
    # 2) gather top-activating patches --------------------------------
    paths = [str(p) for p in Path(cfg.image_dir).glob("*") if p.suffix.lower() in [".jpg", ".png"]]
    heap  = collect_top_patches(sae, vit, paths,
                                layer_idx=cfg.layer,
                                device=device,
                                top_k=cfg.top_k,
                                batch_size=32)

    os.makedirs(cfg.out_dir, exist_ok=True)
    


    # 3) visualise + label --------------------------------------------
    summary = {}
    for unit, lst in heap.items():
        tag = None
        if cfg.auto_label:
            tag = label_unit_clip(lst, device)
        montage_path = f"{cfg.out_dir}/unit{unit:03d}_{tag or 'concept'}.jpg"
        montage(lst, montage_path)
        summary[unit] = {
            "label" : tag,
            "montage": montage_path,
            "patches": [(s, p, idx) for s, p, idx in lst]
        }
        print(f"unit {unit:3d} → {tag or '-'}  saved {montage_path}")

    # 4) stash metadata as json ---------------------------------------
    with open(f"{cfg.out_dir}/summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[✓] everything in  →  {cfg.out_dir}")
    
    metrics_path = f"{cfg.out_dir}/metrics.json"
    with open(metrics_path, "w") as fp:
        json.dump({"mse": mse, "sparsity": spars}, fp, indent=2)
    print(f"[✓] metrics saved  →  {metrics_path}")



# ---------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to SAE *.pt file")
    p.add_argument("--image_dir", required=True, help="folder with new images")
    p.add_argument("--layer", type=int, default=11, help="ViT layer index")
    p.add_argument("--hidden", type=int, default=2048, help="SAE hidden dim")
    p.add_argument("--l1", type=float, default=1e-3, help="λ for the L1 term (only used when re-initialising)")
    p.add_argument("--top_k", type=int, default=30, help="#patches to keep per unit")
    p.add_argument("--out_dir", default="sae_inspect", help="where to put results")
    p.add_argument("--auto_label", action="store_true", help="try to name units with CLIP")
    cfg = p.parse_args()
    main(cfg)
