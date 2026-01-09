import sys
import os

# Add the directory containing chefer_explain to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "explainability-methods-various-models-experiments"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import timm
from torchvision import transforms
from PIL import Image
import medmnist
from medmnist import INFO
from collections import OrderedDict
import math

# Import Chefer LRP
try:
    from chefer_explain.baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
    from chefer_explain.baselines.ViT.ViT_LRP import compute_rollout_attention
except ImportError:
    print("Error: Could not import chefer_explain. Make sure the path is correct.")
    # Fallback/Debug print
    print(f"Current sys.path: {sys.path}")
    exit(1)

# --- Configuration ---
DATA_FLAG = 'pathmnist'
DATA_PATH = "./data"
MODEL_PATH = "best_model.pth"
RESULTS_DIR = "results_pathmnist"
BATCH_SIZE = 1
NUM_SAMPLES = 5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 1. Load Data ---
info = INFO[DATA_FLAG]
n_classes = len(info['label'])
label_dict = info['label']

data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load Test Dataset
TestClass = getattr(medmnist, info['python_class'])
test_dataset = TestClass(split='test', transform=data_transform, download=True, root=DATA_PATH)

# --- 2. Load Timm Model (Finetuned) ---
print("Loading standard ViT model...")
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.head = nn.Linear(in_features=768, out_features=n_classes, bias=True)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

# --- 3. Hooks for Attention Rollout (Timm Model) ---
attention_maps_od = OrderedDict()
q_embeddings_od = OrderedDict()
k_embeddings_od = OrderedDict()
hook_handles = []

def get_query_activations(name:str, q_scale:float):
  def query_activation_hook(module, input, output):
    q_embeddings_od[name] = output.detach() * q_scale
  return query_activation_hook

def get_key_activations(name:str):
  def key_activation_hook(module, input, output):
    k_embeddings_od[name] = output.detach()
  return key_activation_hook

def get_attention_maps(name:str):
  def compute_attn_map_hook(module, input, output):
    q = q_embeddings_od[name]
    k = k_embeddings_od[name]
    attn = q @ k.transpose(-2, -1)
    attention_maps_od[name] = attn.softmax(dim=-1)
  return compute_attn_map_hook

# Register hooks
for i, block in enumerate(model.blocks):
  attn_block = block.attn
  # Hook Q and K norms
  h1 = attn_block.q_norm.register_forward_hook(get_query_activations(f"block_{i}", attn_block.scale))
  h2 = attn_block.k_norm.register_forward_hook(get_key_activations(f"block_{i}"))
  hook_handles.append(h1)
  hook_handles.append(h2)
  
  # Hook MLP to trigger attention map computation (executed after attention block)
  h3 = block.mlp.register_forward_hook(get_attention_maps(f"block_{i}"))
  hook_handles.append(h3)

# --- 4. Setup LRP Model (CheferCAM) ---
print("Initializing LRP model & Transferring weights...")
model_lrp = vit_LRP(pretrained=False, num_classes=n_classes).to(DEVICE)
model_lrp.eval()

# Transfer weights
timm_state_dict = model.state_dict()
lrp_state_dict = model_lrp.state_dict()
new_state_dict = {}

for key in lrp_state_dict.keys():
    if key in timm_state_dict:
        if lrp_state_dict[key].shape != timm_state_dict[key].shape:
            # Handle potential shape mismatch (e.g. pos_embed) if any
            # For standard ViT 224, they should match usually.
            print(f"Shape mismatch for {key}: {lrp_state_dict[key].shape} vs {timm_state_dict[key].shape}")
            if key == 'pos_embed':
                 
                 # Assuming direct copy works for now as both are patch16_224
                 pass
        new_state_dict[key] = timm_state_dict[key]
    else:
        # Handle head naming difference?
        if key.startswith('head') and 'head' not in timm_state_dict:
             # Timm might use 'fc' or similar? No, timm uses 'head'.
             print(f"Warning: {key} not found in timm model")
             pass
        else:
             new_state_dict[key] = timm_state_dict[key]

msg = model_lrp.load_state_dict(new_state_dict, strict=False)
print(f"LRP Model Load Result: {msg}")

# --- 5. Visualization Helper ---
def visualize_overlay(x_tensor, heatmap, save_path, title, alpha=0.6, cmap='jet'):
    # x_tensor: (3, 224, 224) normalized
    # heatmap: (14, 14) or (224, 224) numpy
    
    # Denormalize Image
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(x_tensor.device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(x_tensor.device)
    x_img = x_tensor * std + mean
    img = x_img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    
    # Resize Heatmap
    if heatmap.shape[0] != 224:
        heatmap = cv2.resize(heatmap, (224, 224))
        
    # Normalize Heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(heatmap, cmap=cmap)
    axes[1].set_title("Heatmap")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(heatmap, cmap=cmap, alpha=alpha)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# --- 6. Main Loop ---
NUM_SAMPLES = 500  # As per finetune_vit_base.py subset
print(f"Generating explanations for {NUM_SAMPLES} samples...")

# Function to create the composite plot
def save_composite_plot(img_tensor, target_label, pred_label, pred_prob, all_probs, rollout_map, cam_map, idx, save_path):
    # img_tensor: (3, 224, 224) normalized
    
    # Denormalize Image
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(img_tensor.device)
    x_img = img_tensor * std + mean
    img_np = x_img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    
    # Normalize Heatmaps (0-1)
    def norm_map(m):
        if m.shape[0] != 224:
            m = cv2.resize(m, (224, 224))
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        return m
        
    rollout_map = norm_map(rollout_map)
    cam_map = norm_map(cam_map)
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # 1. Original Image
    axes[0].imshow(img_np)
    axes[0].set_title(f"Sample {idx}\nTrue: {target_label}", fontsize=12)
    axes[0].axis('off')
    
    # 2. Predictions Bar Chart
    class_names = list(label_dict.values())
    probs = all_probs.detach().cpu().numpy().flatten()
    
    # Colors: Green for max, Gray for others
    colors = ['green' if p == probs.max() else 'gray' for p in probs]
    
    y_pos = np.arange(len(class_names))
    axes[1].barh(y_pos, probs, align='center', color=colors)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(class_names)
    axes[1].invert_yaxis()  # labels read top-to-bottom
    axes[1].set_xlabel('Probability')
    axes[1].set_title(f"Predictions (Pred: {pred_label} {pred_prob:.2f})", fontsize=10)
    
    # 3. CheferCAM
    axes[2].imshow(img_np)
    axes[2].imshow(cam_map, cmap='jet', alpha=0.6)
    axes[2].set_title(f"CheferCAM\nExplaining: '{pred_label}'", fontsize=12)
    axes[2].axis('off')
    
    # 4. Attention Rollout
    axes[3].imshow(img_np)
    axes[3].imshow(rollout_map, cmap='jet', alpha=0.6)
    axes[3].set_title("Attention Rollout\n(Class Agnostic)", fontsize=12)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

for idx in range(NUM_SAMPLES):
    if idx >= len(test_dataset): break
    
    img, target = test_dataset[idx] 
    target_cls = target.item()
    target_label = label_dict[str(target_cls)]
    img_batch = img.unsqueeze(0).to(DEVICE)
    
    # --- Attention Rollout & Predictions ---
    # Forward pass (timm)
    output = model(img_batch)
    probs = torch.softmax(output, dim=1)
    pred_cls = probs.argmax(dim=1).item()
    pred_prob = probs[0, pred_cls].item()
    pred_label = label_dict[str(pred_cls)]
    
    # Collect attentions for Rollout
    attn_cams = []
    for i in range(len(model.blocks)):
        name = f"block_{i}"
        attn_tensor = attention_maps_od[name] 
        avg_heads = attn_tensor.mean(dim=1) 
        attn_cams.append(avg_heads)
    
    rollout = compute_rollout_attention(attn_cams, start_layer=0)
    rollout_attn = rollout[:, 0, 1:] 
    rollout_grid = rollout_attn.reshape(14, 14).detach().cpu().numpy()
    
    # --- CheferCAM ---
    model_lrp.zero_grad()
    output_lrp = model_lrp(img_batch)
    
    score = output_lrp[0, pred_cls]
    score.backward()
    
    one_hot = torch.zeros((1, output_lrp.shape[-1]), device=DEVICE)
    one_hot[0, pred_cls] = score
    
    cam_lrp = model_lrp.relprop(one_hot, method="transformer_attribution", alpha=1)
    cam_lrp = cam_lrp.detach().reshape(14, 14).cpu().numpy()
    
    # --- Save Plot ---
    save_name = f"{RESULTS_DIR}/sample_{idx}_{target_label}_{pred_label}.png"
    save_composite_plot(
        img.to(DEVICE), target_label, pred_label, pred_prob, probs, 
        rollout_grid, cam_lrp, idx, save_name
    )
    
    if idx % 10 == 0:
        print(f"Processed {idx}/{NUM_SAMPLES}")

# Cleanup hooks
for h in hook_handles:
    h.remove()

print("Done!")
