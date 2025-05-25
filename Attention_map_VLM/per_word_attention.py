#!/usr/bin/env python3
import os
import math
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import numpy as np
import torchvision.transforms as T

def load_models(device):
    """Load Eagle2-2B, return raw ViT, raw LM, and tokenizer."""
    mm = AutoModel.from_pretrained(
        "nvidia/Eagle2-2B",
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).eval().to(device)
    vit = mm.vision_model.vision_model.to(device)  # SigLIP ViT
    txt = mm.language_model.to(device)            # Qwen LM
    tok = AutoTokenizer.from_pretrained(
        "nvidia/Eagle2-2B",
        trust_remote_code=True
    )
    return vit, txt, tok

def preprocess_image(path, device):
    """Load & resize to 448×448, normalize to SigLIP [-1,1]."""
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    tensor = transform(img).unsqueeze(0).to(device, dtype=torch.float16)
    return tensor, img

def save_heatmap(grid, orig, outpath, color=(0,255,0), threshold=0.5):
    """Overlay thresholded grid as a semi-transparent heatmap on orig."""
    grid_thr = grid.copy()
    grid_thr[grid_thr < threshold] = 0
    hm = Image.fromarray((grid_thr*255).astype(np.uint8), mode='L')
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

def compute_token_similarity(vision_feat, text_hidden, device):
    """Calculate cosine similarity between vision features and each text token."""
    # Handle dimension mismatch
    txt_dim = text_hidden.shape[-1]
    vis_dim = vision_feat.shape[-1]
    
    if txt_dim != vis_dim:
        print(f"Dimension mismatch: text={txt_dim}, vision={vis_dim}. Projecting...")
        with torch.no_grad():
            projection = torch.randn(txt_dim, vis_dim, device=device, dtype=text_hidden.dtype)
            projection = F.normalize(projection, dim=0)  # Orthogonal projection
            text_hidden = text_hidden @ projection
    
    # Normalize features for cosine similarity
    vision_feat_norm = F.normalize(vision_feat, dim=1)  # [P, D]
    text_hidden_norm = F.normalize(text_hidden, dim=1)  # [T, D]
    
    # Compute similarity matrix: each patch × each token
    similarity = vision_feat_norm @ text_hidden_norm.T  # [P, T]
    
    return similarity

def main():
    parser = argparse.ArgumentParser(
        description="Per-word cosine-similarity heatmaps for Eagle2-2B"
    )
    parser.add_argument('--image',      required=True,  help='Input image path')
    parser.add_argument('--prompt',     required=True,  help='Text prompt')
    parser.add_argument('--output_dir', default='cross_attn_outputs',
                        help='Directory to save heatmaps')
    parser.add_argument('--threshold',  type=float, default=0.6,
                        help='Overlay threshold (0–1)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models and tokenizer
    vit, txt, tok = load_models(device)

    # Preprocess image
    img_t, orig_img = preprocess_image(args.image, device)

    # Tokenize and get text hidden states
    text_tokens = tok(
        args.prompt, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    
    with torch.no_grad():
        text_out = txt(
            **text_tokens,
            output_hidden_states=True,
            return_dict=True
        )
        
    # Check if hidden states are available
    if not hasattr(text_out, 'hidden_states') or text_out.hidden_states is None:
        print("Warning: Model didn't return hidden_states, trying to use last_hidden_state")
        if hasattr(text_out, 'last_hidden_state'):
            text_hiddens = text_out.last_hidden_state[0]  # [T, D]
        else:
            raise RuntimeError("Model did not return hidden_states or last_hidden_state")
    else:
        # Get last-layer per-token embeddings [T, D]
        text_hiddens = text_out.hidden_states[-1][0]  # [T, D]
    
    # Get token strings for filenames
    token_strs = tok.convert_ids_to_tokens(text_tokens.input_ids[0])
    print(f"Processing {len(token_strs)} tokens: {token_strs}")
    
    # Run vision backbone to get patch features for each layer
    with torch.no_grad():
        vis_out = vit(
            pixel_values=img_t,
            output_hidden_states=True,
            return_dict=True
        )
    
    if not hasattr(vis_out, 'hidden_states') or vis_out.hidden_states is None:
        raise RuntimeError("Vision model did not return hidden_states")
    
    vis_states = vis_out.hidden_states  # list: [0] embeddings, [1..L] layer outputs
    
    # Process each vision layer (skip embedding layer at index 0)
    for layer_idx in range(1, len(vis_states)):
        print(f"Processing vision layer {layer_idx-1}")
        
        # Get vision features from this layer
        vision_feat = vis_states[layer_idx][0]  # [N, D]
        
        # Remove CLS token if present (compare embedding shape with layer shape)
        if vis_states[0].shape[1] != vis_states[layer_idx].shape[1]:
            vision_feat = vision_feat[1:]
            print(f"Removed CLS token: shape now {vision_feat.shape}")
        
        # Get number of patches and calculate grid size
        P, _ = vision_feat.shape
        g = int(math.sqrt(P))
        if g*g != P:
            print(f"Layer {layer_idx}: cannot reshape {P} patches → square grid, skipping")
            continue
        
        # Compute similarity between vision features and each text token
        similarity = compute_token_similarity(vision_feat, text_hiddens, device)
        
        # For each token, create and save a heatmap
        for tok_idx, tok_str in enumerate(token_strs):
            # Extract similarity for this token
            sim = similarity[:, tok_idx].cpu().numpy()  # [P]
            
            # Normalize similarity scores for visualization
            sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
            
            # Reshape to grid
            grid = sim.reshape(g, g)
            
            # Clean token string for filename
            clean = tok_str.strip().replace("##", "").replace(" ", "_")
            if clean == "":
                clean = f"tok{tok_idx}"
                
            # Save heatmap
            out_fname = f"layer{layer_idx-1:02d}_word_{clean}.png"
            out_path = os.path.join(args.output_dir, out_fname)
            save_heatmap(
                grid, 
                orig_img, 
                out_path,
                color=(0, 255, 0), 
                threshold=args.threshold
            )
            print(f"Saved {out_fname}")

    print(f"✅ Done! Heatmaps in '{args.output_dir}/'")

if __name__ == "__main__":
    main()
