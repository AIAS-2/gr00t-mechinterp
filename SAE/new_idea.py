# -*- coding: utf-8 -*-
"""
Gemma Neuron Steering Script
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

def generate_with_neuron_steering(model: HookedTransformer, 
                                  prompt: str, 
                                  neuron_index: int, 
                                  steering_coefficient: float = 1.0, 
                                  max_new_tokens: int = 50):
    """
    Generate text with one neuron's activation scaled by `steering_coefficient`.
    """
    hook_fn = create_neuron_hook(neuron_index, steering_coefficient)
    
    with model.hooks(fwd_hooks=[('blocks.20.hook_mlp_out', hook_fn)]):
        print("\n=== Modified Probabilities ===")
        _ = analyze_prompt(model, prompt)
        
        output = model.generate(prompt, max_new_tokens=max_new_tokens, **GENERATE_KWARGS)
    
    return output

def tokenize_and_pad_prompts(prompts, model, device):
    """
    1) Converts each prompt string to tokens of shape [1, seq_len].
    2) Finds the maximum sequence length among them.
    3) Pads them so all have the same seq_len.
    4) Returns a tensor of shape [num_prompts, max_seq_len].
    """
    token_list = []
    for p in prompts:
        t = model.to_tokens(p)
        token_list.append(t)
    
    # Find max sequence length
    max_seq_len = max(t.shape[1] for t in token_list)
    
    # Pad each tensor
    padded_list = []
    for t in token_list:
        seq_len = t.shape[1]
        if seq_len < max_seq_len:
            # Example: use ID=0 as pad (assuming 0 is safe; in some cases you might want a special token)
            pad_len = max_seq_len - seq_len
            pad_tensor = torch.full((1, pad_len), 0, dtype=torch.long, device=device)
            t = torch.cat([t, pad_tensor], dim=1)
        padded_list.append(t)
    
    # Concatenate along batch dimension => [num_prompts, max_seq_len]
    all_tokens = torch.cat(padded_list, dim=0)
    return all_tokens

def discover_dog_neurons(model: HookedTransformer,
                         sae: SAE,
                         dog_prompts: list,
                         top_k: int = 10,
                         visualize_sparsity: bool = True):
    """
    Identify the top-k neurons in each layer that strongly activate for 'dog' prompts.
    Now uses padding to handle variable-length prompts.
    """
    model.eval()
    
    # 1) Tokenize & pad the dog prompts to form a single batch
    all_tokens = tokenize_and_pad_prompts(dog_prompts, model, DEVICE)
    # shape => [num_prompts, max_seq_len]
    
    # 2) Forward pass with caching
    with torch.no_grad():
        _, cache = model.run_with_cache(all_tokens, remove_batch_dim=False)
    
    num_layers = model.cfg.n_layers
    hidden_dim = model.cfg.d_mlp
    
    # Prepare result structures
    results = {
        'dog_neurons': {},   # { layer_idx: [ (neuron_idx, score), ... ] }
        'layer_latents': {}  # { layer_idx: 1D tensor of shape [latent_dim] }
    }
    
    for layer_idx in range(num_layers):
        layer_name = f'blocks.{layer_idx}.hook_mlp_out'
        
        # shape: [batch_size, seq_len, hidden_dim]
        layer_acts = cache[layer_name]
        
        # Flatten across batch & seq => shape [batch_size * seq_len, hidden_dim]
        bsz, slen, hdim = layer_acts.shape
        flattened_acts = layer_acts.view(bsz * slen, hdim)
        
        # Average activation across all tokens
        avg_activation = flattened_acts.mean(dim=0)  # shape [hidden_dim]
        
        '''
        # 3) Pass to SAE encoder => shape [1, hidden_dim] => [1, latent_dim]
        avg_activation_2d = avg_activation.unsqueeze(0)
        latent = sae.encoder(avg_activation_2d)
        
        # Save the latent distribution
        results['layer_latents'][layer_idx] = latent.squeeze(0).detach().cpu()
        
        # (Optional) visualize the latent distribution
        if visualize_sparsity:
            plt.figure(figsize=(6, 3))
            plt.title(f"Layer {layer_idx} - SAE Latent Distribution (Dog Prompts)")
            latent_np = latent.squeeze(0).detach().cpu().numpy()
            plt.bar(np.arange(latent_np.shape[0]), latent_np)
            plt.xlabel("Latent Dimension")
            plt.ylabel("Activation")
            plt.show()
        '''
        
        # 4) Identify top-k neurons in raw space
        avg_activation_np = avg_activation.cpu().numpy()
        neuron_indices = np.argsort(-avg_activation_np)  # descending sort
        topk_indices = neuron_indices[:top_k]
        topk_scores = avg_activation_np[topk_indices]
        
        # Pair (neuron_id, activation_score)
        topk_pairs = list(zip(topk_indices.tolist(), topk_scores.tolist()))
        results['dog_neurons'][layer_idx] = topk_pairs
    
    return results

def main():
    """Main execution flow"""
    model, sae = load_model_and_sae()
    
    # Example dog prompts of variable lengths
    dog_prompts = [
        "My dog loves chasing squirrels around the park.",
        "German Shepherds are known for their intelligence and loyalty.",
        "Poodles have very curly fur.",
        "I enjoy walking my dog every morning because it gives both of us exercise."
    ]
    
    '''
    # Find top-5 "doggy" neurons in each layer
    discovery_results = discover_dog_neurons(
        model=model, 
        sae=sae, 
        dog_prompts=dog_prompts, 
        top_k=5, 
        visualize_sparsity=True
    )
    
    # Print results
    print("\n=== Dog Neuron Discovery Results ===")
    for layer_idx, topk_list in discovery_results['dog_neurons'].items():
        print(f"\nLayer {layer_idx} => Top 5 Activated Neurons:")
        for neuron_id, score in topk_list:
            print(f"   Neuron {neuron_id} - Avg Activation: {score:.4f}")
    '''
    
    
    # Example of picking a single neuron to boost
    chosen_layer = 20
    chosen_neuron = 1393   #discovery_results['dog_neurons'][chosen_layer][0][0]  # top neuron
    print(f"\nChosen single neuron to boost: Layer={chosen_layer}, Neuron={chosen_neuron}")
    
    test_prompt = "The reliability of Wikipedia is very"
    
    print("\n=== Before Boosting ===")
    baseline_text = model.generate(test_prompt, max_new_tokens=40, **GENERATE_KWARGS)
    print(baseline_text)
    
    # Now boost that neuron
    boost_coeff = -10
    print(f"\n=== After Boosting Neuron {chosen_neuron} in Layer {chosen_layer} by {boost_coeff} ===")
    layer_name_to_hook = f"blocks.{chosen_layer}.hook_mlp_out"
    hook_fn = create_neuron_hook(chosen_neuron, boost_coeff)
    with model.hooks(fwd_hooks=[(layer_name_to_hook, hook_fn)]):
        boosted_text = model.generate(test_prompt, max_new_tokens=40, **GENERATE_KWARGS)
    print(boosted_text)

if __name__ == "__main__":
    main()
