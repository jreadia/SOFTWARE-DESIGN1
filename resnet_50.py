import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# --- Configuration ---
DATASET_DIR = "Rust_Dataset"
IMG_HEIGHT, IMG_WIDTH = 640, 640
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 4
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def calculate_code_metrics():
    """Calculates code complexity metrics using 'radon'."""
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
        print("[WARN] 'radon' not installed. Skipping Functionality & Compatibility metrics.")
        return 0, 0

def calculate_map(model, dataloader, device):
    """Calculates Mean Average Precision using 'scikit-learn'."""
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
        
        y_pred_probs = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
        return average_precision_score(y_true_bin, y_pred_probs, average="macro")
    except ImportError:
        print("[WARN] 'scikit-learn' not installed. Skipping Performance (mAP) metric.")
        return 0.0

def main():
    print(f"--- Starting ResNet50 Training & Analysis (PyTorch) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=2)
                   for x in ['train', 'valid']}
    
    test_loader = DataLoader(image_datasets['test'], batch_size=1, shuffle=False, num_workers=0)

    # 2. Model Construction
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    for param in model.parameters():
        param.requires_grad = False
        
    # ResNet50 uses 'fc' as the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    print("Training model...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    # --- METRICS CALCULATION ---
    print("\n--- Calculating Metrics ---")

    # A. Manufacturability
    total_params = sum(p.numel() for p in model.parameters())
    
    # B. Efficiency
    model.eval()
    start_time = time.time()
    steps_to_test = 50
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= steps_to_test: break
            inputs = inputs.to(device)
            _ = model(inputs)
    end_time = time.time()
    avg_inference_time_ms = ((end_time - start_time) / steps_to_test) * 1000

    # C. Performance (mAP)
    mAP = calculate_map(model, test_loader, device)

    # D & E. Functionality & Compatibility
    avg_cc, maint_index = calculate_code_metrics()

    # --- FINAL REPORT ---
    print("\n" + "="*50)
    print("RUST DETECTION MODEL: RESNET50 (PyTorch)")
    print("="*50)
    print(f"1. Manufacturability (Model Complexity): {total_params:,} parameters")
    print(f"2. Efficiency (Avg Inference Time):      {avg_inference_time_ms:.2f} ms/image")
    print(f"3. Performance (Mean Average Precision): {mAP:.4f}")
    print(f"4. Functionality (Cyclomatic Complexity):{avg_cc:.2f} (Avg per block)")
    print(f"5. Compatibility (Maintainability Index):{maint_index:.2f} (Scale 0-100)")
    print("="*50)
    
    torch.save(model.state_dict(), 'rust_resnet50.pth')
    print("Model saved to rust_resnet50.pth")

if __name__ == "__main__":
    main()
