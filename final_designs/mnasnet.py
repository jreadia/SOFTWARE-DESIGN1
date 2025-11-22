import os
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# --- Configuration ---
DATASET_DIR = "../Rust_Dataset"
IMG_HEIGHT, IMG_WIDTH = 640, 640
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-4
NUM_CLASSES = 4
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def force_print(msg):
    """Helper to force printing to console immediately."""
    print(msg, flush=True)

def calculate_code_metrics():
    try:
        import radon.complexity as cc
        import radon.metrics as mi
        with open(__file__, 'r') as f:
            code = f.read()
        complexity_data = cc.cc_visit(code)
        avg_cc = np.mean([item.complexity for item in complexity_data]) if complexity_data else 0
        maintainability_index = mi.mi_visit(code, multi=False)
        return avg_cc, maintainability_index
    except ImportError:
        return 0, 0

def calculate_map(model, dataloader, device):
    try:
        from sklearn.metrics import average_precision_score
        from sklearn.preprocessing import label_binarize
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                outputs = torch.softmax(model(inputs), dim=1)
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.numpy())
        if not all_labels: return 0.0
        y_pred_probs = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
        return average_precision_score(y_true_bin, y_pred_probs, average="macro")
    except ImportError:
        return 0.0

def main():
    force_print(f"--- Starting MNASNet Training & Analysis ---")
    
    # Check for dataset existence
    if not os.path.exists(os.path.join(DATASET_DIR, 'train')):
        force_print(f"ERROR: Could not find dataset at '{os.path.abspath(DATASET_DIR)}'")
        force_print("Please ensure the 'train', 'test', and 'valid' folders are in the same directory as this script.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    force_print(f"Using device: {device}")

    # 1. Data Setup
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ]),
        'valid': transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ]),
        'test': transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATASET_DIR, x), data_transforms[x])
                      for x in ['train', 'valid', 'test']}
    
    # IMPORTANT: Changed num_workers to 0 to prevent hanging on Windows/CPUs
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=0)
                   for x in ['train', 'valid']}
    
    test_loader = DataLoader(image_datasets['test'], batch_size=1, shuffle=False, num_workers=0)

    force_print(f"Data loaded. Train images: {len(image_datasets['train'])}")

    # 2. Model Construction (MNASNet 1.0)
    force_print("Downloading/Loading Model...")
    model = models.mnasnet1_0(weights='IMAGENET1K_V1')
    
    # Freeze base
    for param in model.parameters():
        param.requires_grad = False
        
    # MNASNet classifier is model.classifier[1]
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    # 3. Training
    force_print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloaders['train']):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            # Print progress every few batches to ensure user knows it's running
            if i % 5 == 0:
                print(f"\rEpoch {epoch+1} - Batch {i}/{len(dataloaders['train'])}...", end="", flush=True)

        epoch_loss = running_loss / len(image_datasets['train'])
        force_print(f"\nEpoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

    # --- METRICS ---
    force_print("\n--- Calculating Metrics ---")
    total_params = sum(p.numel() for p in model.parameters())
    
    model.eval()
    start_time = time.time()
    steps_to_test = 50
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= steps_to_test: break
            inputs = inputs.to(device)
            _ = model(inputs)
    avg_inference_time_ms = ((time.time() - start_time) / steps_to_test) * 1000

    mAP = calculate_map(model, test_loader, device)
    avg_cc, maint_index = calculate_code_metrics()

    force_print("\n" + "="*50)
    force_print("RUST DETECTION MODEL: MNASNET 1.0 REPORT")
    force_print("="*50)
    force_print(f"1. Manufacturability (Model Complexity): {total_params:,} parameters")
    force_print(f"2. Efficiency (Avg Inference Time):      {avg_inference_time_ms:.2f} ms/image")
    force_print(f"3. Performance (Mean Average Precision): {mAP:.4f}")
    force_print(f"4. Functionality (Cyclomatic Complexity):{avg_cc:.2f}")
    force_print(f"5. Compatibility (Maintainability Index):{maint_index:.2f}")
    force_print("="*50)
    
    torch.save(model.state_dict(), 'rust_mnasnet.pth')
    force_print("Saved model to rust_mnasnet.pth")

if __name__ == "__main__":
    main()
