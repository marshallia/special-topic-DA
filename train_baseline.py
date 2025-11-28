# train_baseline.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm

# --------------------------
# Configuration
# --------------------------
DATA_DIR = 'data/train'       # Path to simulated training data
VAL_DIR = 'data/validation'          # Path to simulated validation data
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_CLASSES = len(os.listdir(DATA_DIR))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
MODEL_SAVE_PATH = 'model/baseline_resnet50.pth'

# --------------------------
# Data Transforms
# --------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # standard for ResNet
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------------------------
# Datasets and Loaders
# --------------------------
train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

# --------------------------
# Model
# --------------------------
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# --------------------------
# Loss & Optimizer
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --------------------------
# Training Loop
# --------------------------
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
train_losses =[]
train_accuracies = []
val_losses =[]
val_accuracies = []
for epoch in range(0, NUM_EPOCHS):
    # Training
    model.train()
    running_loss = 0.0
    running_corrects = 0
    batch_number = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for inputs, labels in progress_bar:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        batch_number += 1
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
    
    # Validation
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)
    
    val_loss = val_running_loss / len(val_dataset)
    val_acc = val_running_corrects.double() / len(val_dataset)
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Best model saved with Val Acc: {best_acc:.4f}")
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

print(f"\nTraining complete. Best Val Acc: {best_acc:.4f}")

# Visualization
epochs = range(1, NUM_EPOCHS + 1)
plt.figure(figsize=(12,5))

# Loss subplot
plt.subplot(1,2,1)
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.plot(epochs, val_losses, 'orange', label='Validation Loss')
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy subplot
plt.subplot(1,2,2)
plt.plot(epochs, train_accuracies, 'g-', label='Training Accuracy')
plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
plt.title('Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()