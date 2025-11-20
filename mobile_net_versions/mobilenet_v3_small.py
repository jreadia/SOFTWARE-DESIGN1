import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import radon.complexity as cc
import radon.metrics as mi

# --- CONFIGURATION ---
DATASET_DIR = "../Rust_Dataset"
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.001
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATASET_DIR, x), data_transforms[x])
                      for x in ['train', 'valid', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=2)
                   for x in ['train', 'valid']}
    test_loader = DataLoader(image_datasets['test'], batch_size=1, shuffle=False)
    return dataloaders, test_loader

def main():
    print(f"--- Training MobileNetV3-Small on {DEVICE} ---")
    dataloaders, test_loader = get_dataloaders()

    # Build Model
    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    
    # V3 Head: classifier[3] is the Linear layer
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Acc: {correct/total:.4f}")

    # Gather Metrics
    print("\n--- Results for MobileNetV3-Small ---")
    
    # 1. Manufacturability
    total_params = sum(p.numel() for p in model.parameters())
    
    # 2. Efficiency
    model.eval()
    start_time = time.time()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    end_time = time.time()
    inf_time = ((end_time - start_time) / len(test_loader.dataset)) * 1000
    
    # 3. Performance
    y_true = np.concatenate(all_labels)
    y_scores = np.concatenate(all_probs)
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    map_score = average_precision_score(y_true_bin, y_scores, average='macro')
    
    # 4 & 5. Code Metrics
    try:
        with open(__file__, 'r') as f: code = f.read()
        cc_score = np.mean([i.complexity for i in cc.cc_visit(code)])
        mi_score = mi.mi_visit(code, multi=False)
    except: cc_score, mi_score = 0, 0

    print(f"1. Manufacturability (Params):      {total_params:,}")
    print(f"2. Efficiency (Inference Time):     {inf_time:.2f} ms/image")
    print(f"3. Performance (mAP):               {map_score:.4f}")
    print(f"4. Functionality (Cyclomatic Cplx): {cc_score:.2f}")
    print(f"5. Compatibility (Maint. Index):    {mi_score:.2f}")
    
    torch.save(model.state_dict(), 'rust_mobilenetv3_small.pth')

if __name__ == "__main__":
    main()