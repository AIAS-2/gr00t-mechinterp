"""
Gemma Neuron Steering Script with SAE
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformer_lens import HookedTransformer, utils
from sae_lens import SAE
from functools import partial

# Configuration
MODEL_NAME = "gemma-2-2b-it"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_20/width_16k/canonical"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # the mps device doesn't work on my mac
GENERATE_KWARGS = dict(temperature=0.5, freq_penalty=2.0, verbose=False)


def load_model_and_sae():
    """Load the Gemma model and associated Sparse Autoencoder"""
    print("Loading model and SAE...")
    
    # Load instruction-tuned Gemma model
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    
    # Load pre-trained SAE for model analysis
    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=DEVICE)[0]
    
    return model, sae


def analyze_prompt(model, prompt: str):
    """Analyze model's response to a prompt and return probabilities + cache"""
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    
    # Calculate probabilities for the last token
    probs = logits[:, -1, :].softmax(dim=-1)
    token_probs, token_ids = probs.topk(10, dim=-1)
    
    print("\nTop predicted tokens:")
    for tk, p in zip(token_ids[0], token_probs[0]):
        print(f"{model.to_string(tk.item())}: {p.item():.4f}")
    
    return cache

    """Visualize neural activations using matplotlib"""
    plt.figure(figsize=(8, 6))
    activation_data = activations.detach().cpu().numpy()[-1, :]
    plt.imshow(activation_data.reshape(48, 48))
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()

def create_neuron_hook(neuron_index: int, steering_coefficient: float):
    """Create hook function to modify specific neuron's output"""
    def neuron_hook(value, hook):
        value[:, :, neuron_index] *= steering_coefficient
        return value
    return neuron_hook


def sae_hook(activations: torch.Tensor, hook, sae: SAE, specific_feature_idx: int, boost: float):
    """
    Boosts the `specifc_feature_idx` in the SAE latent space.
    """
    # Encode activations into SAE latents
    latents = sae.encode(activations)  # [batch, seq, sae_dim]
    
    # Boost the specific feature
    latents[:, :, specific_feature_idx] += boost
    
    # Decode back to model activation space
    modified_activations = sae.decode(latents)
    
    return modified_activations



def main():

    model, sae = load_model_and_sae()
    
    # Dog-related SAE feature (from discovery)
    FEATURE_THEME = "DOG"
    SPECIFIC_FEATURE_IDX = 12082  # From Neuronpedia/gemma-scope-res-16k - this one is about dog-related stuff for now
    LAYER = 20
    BOOST = 275
    
    
    prompt = "Tell me a story."
    
    
    # Normal output of the model
    initial_output = model.generate(prompt, max_new_tokens=50, **GENERATE_KWARGS)
    print("\n===== Baseline Output =====")
    print(f"{initial_output}\n\n")
    
    
    # Generate with SAE steering
    with model.hooks(fwd_hooks=[
        (f"blocks.{LAYER}.hook_resid_post", 
         partial(sae_hook, sae=sae, specific_feature_idx=SPECIFIC_FEATURE_IDX, boost=BOOST))
    ]):
        output = model.generate(prompt, max_new_tokens=50, **GENERATE_KWARGS)
    
    print(f"===== {str(FEATURE_THEME)}-Steered Output =====")
    print(output)

if __name__ == "__main__":
    main()
