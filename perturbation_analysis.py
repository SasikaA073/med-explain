import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import medmnist
from medmnist import INFO
import timm
from tqdm import tqdm

# Add the directory containing chefer_explain to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "explainability-methods-various-models-experiments"))

# Import Chefer LRP
try:
    from chefer_explain.baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
except ImportError:
    print("Error: Could not import chefer_explain.")
    sys.exit(1)

# --- Configuration ---
DATA_FLAG = 'pathmnist'
DATA_PATH = "./data"
MODEL_PATH = "best_model.pth"
RESULTS_DIR = "results_pathmnist"
NUM_SAMPLES = 100
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Load Data ---
info = INFO[DATA_FLAG]
n_classes = len(info['label'])
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])
TestClass = getattr(medmnist, info['python_class'])
test_dataset = TestClass(split='test', transform=data_transform, download=True, root=DATA_PATH)

# --- Load Timm Model (for Prediction) ---
print("Loading standard ViT model...")
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.head = nn.Linear(in_features=768, out_features=n_classes, bias=True)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

# --- Load LRP Model (for Explanation) ---
print("Loading LRP model...")
model_lrp = vit_LRP(pretrained=False, num_classes=n_classes).to(DEVICE)
model_lrp.eval()

# Transfer weights
timm_state_dict = model.state_dict()
lrp_state_dict = model_lrp.state_dict()
new_state_dict = {}
for key in lrp_state_dict.keys():
    if key in timm_state_dict:
        new_state_dict[key] = timm_state_dict[key]
    else:
        new_state_dict[key] = timm_state_dict[key] # Simplified copy, assuming compatibility from previous check
model_lrp.load_state_dict(new_state_dict, strict=False)

# --- Perturbation Logic ---

def generate_cam(img_tensor, target_class):
    # img_tensor: (1, 3, 224, 224)
    model_lrp.zero_grad()
    output_lrp = model_lrp(img_tensor)
    score = output_lrp[0, target_class]
    score.backward()
    
    one_hot = torch.zeros((1, output_lrp.shape[-1]), device=DEVICE)
    one_hot[0, target_class] = score
    
    cam = model_lrp.relprop(one_hot, method="transformer_attribution", alpha=1)
    cam = cam.detach().reshape(14, 14) # (14, 14)
    return cam

def perturb_image(img_tensor, mask_indices, patch_size=16):
    # img_tensor: (1, 3, 224, 224)
    # mask_indices: list of (row, col) patch indices to mask
    # Masking with 0 (mean value since normalized)
    
    img_perturbed = img_tensor.clone()
    for (r, c) in mask_indices:
        y_start, y_end = r * patch_size, (r + 1) * patch_size
        x_start, x_end = c * patch_size, (c + 1) * patch_size
        img_perturbed[:, :, y_start:y_end, x_start:x_end] = 0.0
    return img_perturbed

def evaluate_perturbation(samples_dataset, num_samples=100, steps=10):
    percentages = np.linspace(0, 100, steps + 1)
    morf_scores = {p: [] for p in percentages}
    lerf_scores = {p: [] for p in percentages}
    
    print(f"Running perturbation analysis on {num_samples} samples...")
    
    for idx in tqdm(range(num_samples)):
        img, _ = samples_dataset[idx]
        img_batch = img.unsqueeze(0).to(DEVICE)
        
        # Get Initial Prediction
        with torch.no_grad():
            output = model(img_batch)
            probs = torch.softmax(output, dim=1)
            pred_cls = probs.argmax(dim=1).item()
            init_prob = probs[0, pred_cls].item()
        
        # We only explain valid predictions? Or all? Usually Perturbation curve is computed for the PREDICTED class.
        target_cls = pred_cls
        
        # Generate Explanation (Attribution Map)
        try:
            cam = generate_cam(img_batch, target_cls) # (14, 14)
        except Exception as e:
            print(f"Error generating CAM for sample {idx}: {e}")
            continue
            
        # Rank Patches
        cam_flat = cam.flatten() # (196,)
        sorted_indices = torch.argsort(cam_flat, descending=True).cpu().numpy() # Most important first
        
        # Convert flat indices to (row, col)
        patches_rc = [(i // 14, i % 14) for i in sorted_indices]
        
        # --- MoRF (Remove Most Important) ---
        for p in percentages:
            k = int((p / 100.0) * 196)
            mask_indices = patches_rc[:k]
            
            img_masked = perturb_image(img_batch, mask_indices)
            
            with torch.no_grad():
                out_masked = model(img_masked)
                prob_masked = torch.softmax(out_masked, dim=1)[0, target_cls].item()
            
            morf_scores[p].append(prob_masked)
            
        # --- LeRF (Remove Least Important) ---
        # Reverse user 'patches_rc' (Least important first)
        patches_rc_lerf = patches_rc[::-1]
        
        for p in percentages:
            k = int((p / 100.0) * 196)
            mask_indices = patches_rc_lerf[:k]
            
            img_masked = perturb_image(img_batch, mask_indices)
            
            with torch.no_grad():
                out_masked = model(img_masked)
                prob_masked = torch.softmax(out_masked, dim=1)[0, target_cls].item()
            
            lerf_scores[p].append(prob_masked)
        
    # Average scores
    avg_morf = [np.mean(morf_scores[p]) for p in percentages]
    avg_lerf = [np.mean(lerf_scores[p]) for p in percentages]
    
    return percentages, avg_morf, avg_lerf

# --- Run Evaluation ---
percentages, avg_morf, avg_lerf = evaluate_perturbation(test_dataset, num_samples=NUM_SAMPLES)

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(percentages, avg_morf, 'r-o', label='MoRF (Remove Important)')
plt.plot(percentages, avg_lerf, 'b-o', label='LeRF (Remove Unimportant)')
plt.title(f'Quantitative Verification: Perturbation Analysis (N={NUM_SAMPLES})')
plt.xlabel('Percentage of Image Masked')
plt.ylabel('Prediction Probability')
plt.grid(True)
plt.legend()
plt.ylim(0, 1.0)
plt.savefig(f"{RESULTS_DIR}/perturbation_analysis.png")
print(f"Perturbation analysis graph saved to {RESULTS_DIR}/perturbation_analysis.png")
