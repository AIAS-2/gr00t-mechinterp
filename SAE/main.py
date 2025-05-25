# -*- coding: utf-8 -*-
"""
Gemma Neuron Steering Script

A script to analyze and modify neuron behavior in the Gemma-2-2b-it LLM.
Focuses on visualizing activations and steering specific neurons to influence model outputs.
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
SAE_RELEASE = "gemma-scope-2b-pt-mlp-canonical"
SAE_ID = "layer_20/width_16k/canonical"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
    """Analyze model's response to a prompt and return probabilities"""
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    
    # Calculate probabilities for last token
    probs = logits[:, -1, :].softmax(dim=-1)
    token_probs, token_ids = probs.topk(10, dim=-1)
    
    # Print top predictions
    print("\nTop predicted tokens:")
    for tk, p in zip(token_ids[0], token_probs[0]):
        print(f"{model.to_string(tk.item())}: {p.item():.4f}")
    
    return cache

def visualize_activations(activations: torch.Tensor, title: str):
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

def generate_with_neuron_steering(model: HookedTransformer, prompt: str, neuron_index: int, steering_coefficient: float = 1.0, max_new_tokens: int = 50):
    
    
    """Generate text with modified neuron behavior"""
    hook_fn = create_neuron_hook(neuron_index, steering_coefficient)
    
    with model.hooks(fwd_hooks=[('blocks.20.hook_mlp_out', hook_fn)]): #here it only looks at the 20th layer 'aka' output
        # FIRST: Analyze modified probabilities
        print("\n=== Modified Probabilities ===")
        modified_cache = analyze_prompt(model, prompt)
        
        # THEN: Generate text with same modification
        output = model.generate(prompt, max_new_tokens=max_new_tokens, **GENERATE_KWARGS)
    
    return output

def main():
    """Main execution flow"""
    # Initialize components
    model, sae = load_model_and_sae()
    
    # prompt - from the Welch Labs video
    prompt = "The reliability of Wikipedia is very"
    
    # baseline response
    print("\n=== Baseline Analysis ===")
    cache = analyze_prompt(model, prompt)
    
    # Visualize activations at different layers
    #visualize_activations(
    #    cache['hook_embed'],
    #    "Initial Embedding Activations"
    #)
    #visualize_activations(
    #    cache['blocks.0.hook_attn_out'],
    #    "First Attention Layer Output"
    #)
    
    NEURON_INDEX = 1393
    STEERING_COEFFICIENT = -10.0
    
    print(f"\nSteering neuron {NEURON_INDEX} with coefficient {STEERING_COEFFICIENT}")
    
    # Generate with modified neuron behavior
    modified_output = generate_with_neuron_steering(model, prompt, NEURON_INDEX, STEERING_COEFFICIENT)
    
    print("\nModified Output:")
    print(modified_output)


if __name__ == "__main__":
    main()
