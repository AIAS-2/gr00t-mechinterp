import argparse
import os

import torch
from transformers import AutoModel
from PIL import Image
import numpy as np


def load_model(device):
    # Load Eagle2‑2B in FP16
    model = AutoModel.from_pretrained(
        "nvidia/Eagle2-2B",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    return model.to(device)


def preprocess_image(image_path):
    # Load and resize image to 448×448
    img = Image.open(image_path).convert("RGB")
    img = img.resize((448, 448), Image.BICUBIC)

    # Convert to numpy, normalize
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # Convert to tensor [1,C,H,W]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor, img


def feature_inversion(vit, target_feats, layer_idx, device,
                      num_iters=200, lr=0.1):
    """
    Optimize a random image so its features at layer_idx match target_feats.
    vit: SigLIPVisionTransformer
    target_feats: tensor (1, num_patches, dim)
    """
    recon = torch.rand((1, 3, 448, 448), device=device,
                       dtype=torch.float32, requires_grad=True)
    mean = torch.tensor([0.485, 0.456, 0.406],
                        device=device, dtype=torch.float16).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225],
                        device=device, dtype=torch.float16).view(1, 3, 1, 1)
    optimizer = torch.optim.Adam([recon], lr=lr)
    loss_fn = torch.nn.MSELoss()

    for _ in range(num_iters):
        optimizer.zero_grad()
        normed = (recon - mean) / std
        normed = normed.half()
        out = vit(
            pixel_values=normed,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True
        )
        curr = out.hidden_states[layer_idx + 1]
        loss = loss_fn(curr, target_feats)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            recon.clamp_(0.0, 1.0)

    arr = recon.detach().cpu().numpy()[0]
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="input_images/real_dog.png",
                        help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="outputit",
                        help="Where to save results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    img_t, orig = preprocess_image(args.image)
    img_t = img_t.to(device, dtype=torch.float16)

    vit = model.vision_model.vision_model

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    normed = (img_t.float() - mean) / std
    normed = normed.half()

    with torch.no_grad():
        vit_out = vit(
            pixel_values=normed,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )
    hidden_states = list(vit_out.hidden_states)
    attentions    = list(vit_out.attentions)

    os.makedirs(args.output_dir, exist_ok=True)

    for layer_idx in range(len(attentions)):
        layer_dir = os.path.join(args.output_dir, f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)

        if layer_idx < 27:
            target = hidden_states[layer_idx + 1].detach()
            rec_img = feature_inversion(vit, target,
                                        layer_idx, device)
            rec_img.save(os.path.join(
                layer_dir, f"layer{layer_idx}_inversion.png"
            ))
    

    print(f"Outputs saved under {args.output_dir}/layer_*/")
