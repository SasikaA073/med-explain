import argparse
import os

# --- 1. Parse Arguments & Set CUDA Device First ---
parser = argparse.ArgumentParser(description='MedMNIST ViT Finetuning')
parser.add_argument('--cuda_device', type=str, default="0", help='ID of the CUDA device to use (e.g., "0", "0,1")')
args = parser.parse_args()

# Set the environment variable BEFORE importing torch.cuda or initializing devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

import timm 
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator
import wandb
import torch.nn.functional as F

# --- Configuration ---
do_train = False
NUM_EPOCHS = 10
BATCH_SIZE = 128
lr = 0.001
data_flag = 'pathmnist'
download = True
data_path = "./data"
model_save_path = "best_model.pth"

# !!! SAMPLE LIMITS (Set to None to use full dataset) !!!
MAX_TRAIN_SAMPLES = 5000
MAX_VAL_SAMPLES = 500
MAX_TEST_SAMPLES = 500

# Initialize WandB
wandb.init(
    project="medmnist-vit-finetune",
    config={
        "learning_rate": lr,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "model": "vit_base_patch16_224",
        "dataset": data_flag,
        "cuda_device": args.cuda_device,
        "train_samples": MAX_TRAIN_SAMPLES,
        "val_samples": MAX_VAL_SAMPLES
    }
)

# --- Preprocessing ---
data_transform = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

os.makedirs(data_path, exist_ok=True)

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

print(f"--------------------------------")
print(f"Dataset: {data_flag}")
print(f"Task Type: {task}")
print(f"Number of Classes: {n_classes}")
print(f"--------------------------------")

DataClass = getattr(medmnist, info['python_class'])

# Load the full data first
full_train_dataset = DataClass(split='train', transform=data_transform, download=download, root=data_path)
full_val_dataset = DataClass(split='val', transform=data_transform, download=download, root=data_path)
full_test_dataset = DataClass(split='test', transform=data_transform, download=download, root=data_path)

# --- Subsetting Helper Function ---
def limit_dataset(dataset, max_samples):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    indices = np.arange(max_samples)
    return data.Subset(dataset, indices)

# Apply limits
train_dataset = limit_dataset(full_train_dataset, MAX_TRAIN_SAMPLES)
val_dataset = limit_dataset(full_val_dataset, MAX_VAL_SAMPLES)
test_dataset = limit_dataset(full_test_dataset, MAX_TEST_SAMPLES)

print(f"Samples used - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# Create DataLoaders
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Model Setup ---
# Prioritize MPS (Mac), then CUDA, then CPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")
if device == "cuda":
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.head = nn.Linear(in_features=768, out_features=n_classes, bias=True)

trainable_params = []
for name, param in model.named_parameters():
    if 'head' in name:
        param.requires_grad = True
        trainable_params.append(param)
    else:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(trainable_params, lr=lr, momentum=0.9)

model.to(device)

# --- Training Loop ---
best_acc = 0.0

if do_train: 
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            if task != 'multi-label, binary-class':
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    loss = criterion(outputs, targets)
                else:
                    targets = targets.squeeze().long()
                    loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                
                if task != 'multi-label, binary-class':
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_acc,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc
        })

        if val_acc > best_acc:
            print(f"Validation accuracy improved from {best_acc:.2f}% to {val_acc:.2f}%. Saving model...")
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)


# --- Inference with MedMNIST Evaluator ---
print("\nLoading best model for MedMNIST Evaluation...")
model.load_state_dict(torch.load(model_save_path, weights_only=True))
model.eval()

y_score_list = []

with torch.no_grad():
    for inputs, targets in tqdm(test_loader, desc="Testing"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        if task == 'multi-label, binary-class':
            outputs = torch.sigmoid(outputs)
        else:
            outputs = F.softmax(outputs, dim=1)
            
        y_score_list.append(outputs.detach().cpu().numpy())

y_score = np.concatenate(y_score_list, axis=0)

print(f"\nRunning MedMNIST Evaluator for split='test' (Subset size: {len(test_dataset)})...")

evaluator = Evaluator(data_flag, 'test', root=data_path)

# Correctly handle ground truth for subset vs full dataset
if isinstance(test_dataset, torch.utils.data.Subset):
    subset_indices = test_dataset.indices
    evaluator.labels = full_test_dataset.labels[subset_indices]
else:
    evaluator.labels = test_dataset.labels

metrics = evaluator.evaluate(y_score)

print(f"\n--- Final Results ({task}) ---")
if isinstance(metrics, (list, tuple)):
     print(f"AUC: {metrics[0]:.4f}")
     print(f"ACC: {metrics[1]:.4f}")
     wandb.log({"test_auc": metrics[0], "test_acc": metrics[1]})
else:
    print(metrics)
    wandb.log({"test_metrics": metrics})

wandb.finish()