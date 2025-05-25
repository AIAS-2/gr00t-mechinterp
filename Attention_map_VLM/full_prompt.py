import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import numpy as np
import torchvision.transforms as T
import argparse

# ——————————————————————————————
# 1. Helpers: load, preprocess, inversion, rollout, heatmap
# ——————————————————————————————

def load_models(device):
    """Load Eagle2-2B, return raw vision transformer, raw language model, and tokenizer."""
    mm = AutoModel.from_pretrained(
        "nvidia/Eagle2-2B",
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).eval().to(device)
    tok = AutoTokenizer.from_pretrained("nvidia/Eagle2-2B", trust_remote_code=True)
    # raw SigLIP vision transformer
    vit = mm.vision_model.vision_model.to(device)
    # raw language model
    txt = mm.language_model.to(device)
    return vit, txt, tok

def preprocess_image(path, device):
    """Load & resize to model's expected size, normalize with SigLIP [-1,1]."""
    img = Image.open(path).convert("RGB")
    size = getattr(transformers:=None, "None", 448)  # fallback; we override
    # SigLIP uses mean=std=0.5
    transform = T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    t = transform(img).unsqueeze(0).to(device, dtype=torch.float16)
    return t, img

def feature_inversion(vit, target_feats, layer_idx, device, num_iters=200, lr=0.1):
    """
    Optimize random image so its features at `layer_idx` match `target_feats`.
    """
    recon = torch.rand((1,3,448,448), device=device, requires_grad=True, dtype=torch.float32)
    opt = torch.optim.Adam([recon], lr=lr)
    mse = torch.nn.MSELoss()
    mean = torch.tensor([0.5,0.5,0.5],device=device).view(1,3,1,1)
    std  = torch.tensor([0.5,0.5,0.5],device=device).view(1,3,1,1)

    for _ in range(num_iters):
        opt.zero_grad()
        x = recon.clamp(0,1)
        norm = ((x - mean)/std).half()
        # forward up to layer
        h = vit.embeddings(norm)
        b,seq,_ = h.shape
        mask = torch.ones((b,seq),device=device,dtype=torch.long)
        for blk in vit.encoder.layers[:layer_idx+1]:
            h = blk(h,attention_mask=mask,output_attentions=False)[0]
        loss = mse(h, target_feats)
        loss.backward()
        opt.step()

    out = recon.detach().clamp(0,1).cpu()[0].permute(1,2,0).numpy()
    img = (out*255).astype(np.uint8)
    return Image.fromarray(img)

def compute_self_attentions(vit, normed):
    """Manually run embeddings + each block to collect self-attention tensors."""
    h = vit.embeddings(normed)
    b,seq,_ = h.shape
    mask = torch.ones((b,seq),device=h.device,dtype=torch.long)
    atts=[]
    hidden_states = [h]  # Store hidden states as fallback
    for i, blk in enumerate(vit.encoder.layers):
        try:
            out, att = blk(h, attention_mask=mask, output_attentions=True)
            h = out
            atts.append(att)  # shape [1, heads, seq, seq]
            if att is None:
                print(f"Layer {i} attention is None")
        except Exception as e:
            print(f"Error getting attention for layer {i}: {e}")
            h = blk(h, attention_mask=mask, output_attentions=False)
            atts.append(None)
        hidden_states.append(h)
    return atts, hidden_states

def visualize_hidden_state(hidden, orig, outpath, color=(0,0,255)):
    """Visualize hidden state by averaging over channels."""
    # Mean across feature dimension - detach before converting to numpy
    activations = hidden.mean(dim=2).squeeze(0).detach().cpu().numpy()  # Shape: [seq_len]
    
    # Reshape to 2D grid (assuming square grid of patches)
    grid_size = int(np.sqrt(len(activations)))
    grid = activations[:grid_size*grid_size].reshape(grid_size, grid_size)
    
    # Normalize for visualization
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    
    # Use existing save_heatmap function
    save_heatmap(grid, orig, outpath, color=color)

def rollout_matrix(atts):
    """Apply attention‐rollout (add identity, normalize, multiply)."""
    N = atts[0].shape[-1]
    device = atts[0].device
    roll = torch.eye(N,device=device)
    eye  = roll.clone()
    for A in atts:
        m = A[0].mean(dim=0)      # [seq, seq]
        aug = m + eye
        aug = aug/aug.sum(dim=-1,keepdim=True)
        roll = aug @ roll
    return roll

def save_heatmap(grid, orig, outpath, color=(255,0,0)):
    """
    Upsample `grid` (2D numpy in [0,1]) to image size and overlay as `color` heatmap.
    """
    hm = Image.fromarray(np.uint8(grid*255),mode='L')
    hm = hm.resize(orig.size,Image.BILINEAR)
    arr = np.array(hm)/255.0
    overlay = Image.new("RGBA",orig.size)
    ov = overlay.load()
    for y in range(orig.size[1]):
        for x in range(orig.size[0]):
            alpha = int(arr[y,x]*180)
            ov[x,y] = (*color,alpha)
    comp = Image.alpha_composite(orig.convert("RGBA"),overlay)
    comp.save(outpath)

def save_heatmap(grid, orig, outpath, color=(0, 255, 0), threshold=0.5):
    """
    Upsample `grid` (2D numpy in [0,1]) to image size and overlay as heatmap.
    Apply threshold to only show values above threshold.
    """
    # Apply threshold - only show values above threshold
    thresholded_grid = np.copy(grid)
    thresholded_grid[thresholded_grid < threshold] = 0
    
    hm = Image.fromarray(np.uint8(thresholded_grid*255), mode='L')
    hm = hm.resize(orig.size, Image.BILINEAR)
    arr = np.array(hm)/255.0
    overlay = Image.new("RGBA", orig.size)
    ov = overlay.load()
    for y in range(orig.size[1]):
        for x in range(orig.size[0]):
            alpha = int(arr[y, x] * 180)
            ov[x, y] = (*color, alpha)
    comp = Image.alpha_composite(orig.convert("RGBA"), overlay)
    comp.save(outpath)

def prompt_conditioned_similarity(vision_feat, text_emb, device):
    """Calculate cosine similarity between vision features and text embedding."""
    # Handle dimension mismatch
    txt_dim = text_emb.shape[-1]
    vis_dim = vision_feat.shape[-1]
    
    if txt_dim != vis_dim:
        with torch.no_grad():
            projection = torch.randn(txt_dim, vis_dim, device=device, dtype=text_emb.dtype)
            projection = F.normalize(projection, dim=0)
            text_emb = text_emb @ projection
    
    # Calculate similarity
    sim = F.cosine_similarity(
        vision_feat,
        text_emb.repeat(vision_feat.shape[0], 1),
        dim=1
    )
    return sim

# ——————————————————————————————
# 2. Main
# ——————————————————————————————

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--output_dir', default='car', help='Results dir')
    parser.add_argument('--prompt', default='Find the ball', help='Text prompt')
    parser.add_argument('--threshold', type=float, default=0.6, help='Heatmap threshold (0-1)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit, txt, tok = load_models(device)

    # preprocess
    img_t, orig = preprocess_image(args.image, device)

    # Get text embeddings first
    with torch.no_grad():
        text_tokens = tok(args.prompt, return_tensors="pt", padding=True).to(device)
        text_output = txt(**text_tokens, output_hidden_states=True, return_dict=True)
    
    # Get text embedding from last hidden state or hidden_states
    if hasattr(text_output, 'hidden_states') and text_output.hidden_states is not None:
        text_emb = text_output.hidden_states[-1].mean(dim=1)
    elif hasattr(text_output, 'logits'):
        text_emb = text_output.logits.mean(dim=1)
    else:
        raise ValueError("Could not extract text embeddings")

    # Get vision hidden states for all layers
    with torch.no_grad():
        vision_output = vit(
            pixel_values=img_t,
            output_hidden_states=True,
            return_dict=True
        )
    
    hidden_states = vision_output.hidden_states
    
    # Generate prompt-conditioned heatmaps for each layer
    for layer_idx in range(1, len(hidden_states)):
        print(f"Processing layer {layer_idx-1}")
        
        # Get vision features for this layer
        vision_feat = hidden_states[layer_idx][0]
        
        # Check if CLS token present and remove it
        if hidden_states[0].shape[1] != hidden_states[1].shape[1]:
            vision_feat = vision_feat[1:]
        
        # Calculate similarity
        similarity = prompt_conditioned_similarity(vision_feat, text_emb, device)
        
        # Normalize and reshape
        sim_np = similarity.cpu().numpy()
        sim_np = (sim_np - sim_np.min()) / (sim_np.max() - sim_np.min() + 1e-8)
        
        # Calculate grid size (assuming square grid)
        g = int(np.sqrt(len(sim_np)))
        grid = sim_np.reshape(g, g)
        
        # Save heatmap
        save_heatmap(
            grid, 
            orig, 
            os.path.join(args.output_dir, f'layer{layer_idx-1}_prompt.png'), 
            color=(0, 255, 0),
            threshold=args.threshold
        )
    
    print(f'Done. Outputs in {args.output_dir}')
