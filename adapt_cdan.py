# adapt_cdan.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import copy
import numpy as np

# --------------------------
# Configuration
# --------------------------
SIM_DATA_DIR = 'data/sim/train'
REAL_UNLABELED_DIR = 'data/real/unlabeled'
VAL_DIR = 'data/real/val'
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = 'cdan_resnet18.pth'

# --------------------------
# Data Transforms
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --------------------------
# Datasets & Loaders
# --------------------------
sim_dataset = datasets.ImageFolder(SIM_DATA_DIR, transform=transform)
real_dataset = datasets.ImageFolder(REAL_UNLABELED_DIR, transform=transform)  # labels not used
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

sim_loader = DataLoader(sim_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
real_loader = DataLoader(real_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes = sim_dataset.classes
NUM_CLASSES = len(classes)
print(f'Classes: {classes}')

# --------------------------
# Gradient Reversal Layer
# --------------------------
from torch.autograd import Function

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha):
    return GradReverse.apply(x, alpha)

# --------------------------
# CDAN Model
# --------------------------
class CDANModel(nn.Module):
    def __init__(self, num_classes):
        super(CDANModel, self).__init__()
        self.feature = models.resnet18(pretrained=True)
        self.feature.fc = nn.Identity()
        self.class_classifier = nn.Linear(512, num_classes)
        self.domain_classifier = nn.Sequential(
            nn.Linear(512*num_classes, 100),
            nn.ReLU(),
            nn.Linear(100,2)
        )
        
    def forward(self, x, alpha=0):
        feat = self.feature(x)
        class_output = self.class_classifier(feat)
        softmax_out = torch.softmax(class_output, dim=1)
        # Conditional feature
        op = torch.bmm(softmax_out.unsqueeze(2), feat.unsqueeze(1))
        op = op.view(op.size(0), -1)
        reverse_feat = grad_reverse(op, alpha)
        domain_output = self.domain_classifier(reverse_feat)
        return class_output, domain_output

model = CDANModel(NUM_CLASSES).to(DEVICE)

# --------------------------
# Loss & Optimizer
# --------------------------
class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --------------------------
# Training Loop
# --------------------------
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    len_dataloader = min(len(sim_loader), len(real_loader))
    sim_iter = iter(sim_loader)
    real_iter = iter(real_loader)
    
    total_class_loss = 0.0
    total_domain_loss = 0.0
    correct_class = 0
    
    for i in range(len_dataloader):
        p = float(i + epoch * len_dataloader) / (NUM_EPOCHS * len_dataloader)
        alpha = 2. / (1.+np.exp(-10*p)) - 1
        
        # Load batches
        sim_data, sim_label = next(sim_iter)
        real_data, _ = next(real_iter)
        
        sim_data, sim_label = sim_data.to(DEVICE), sim_label.to(DEVICE)
        real_data = real_data.to(DEVICE)
        
        # Combine data
        all_data = torch.cat([sim_data, real_data], 0)
        domain_label = torch.cat([torch.zeros(sim_data.size(0)), torch.ones(real_data.size(0))],0).long().to(DEVICE)
        
        optimizer.zero_grad()
        class_output, domain_output = model(all_data, alpha)
        
        class_loss = class_criterion(class_output[:sim_data.size(0)], sim_label)
        domain_loss = domain_criterion(domain_output, domain_label)
        loss = class_loss + domain_loss
        loss.backward()
        optimizer.step()
        
        total_class_loss += class_loss.item()
        total_domain_loss += domain_loss.item()
        _, preds = torch.max(class_output[:sim_data.size(0)],1)
        correct_class += torch.sum(preds == sim_label.data)
    
    class_acc = correct_class.double() / (len_dataloader*BATCH_SIZE)
    print(f"Epoch {epoch}/{NUM_EPOCHS} | Class Acc: {class_acc:.4f} | Class Loss: {total_class_loss/len_dataloader:.4f} | Domain Loss: {total_domain_loss/len_dataloader:.4f}")
    
    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs,1)
            val_correct += torch.sum(preds == labels.data)
    val_acc = val_correct.double() / len(val_dataset)
    print(f"Validation Acc: {val_acc:.4f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Saved best CDAN model with Val Acc: {best_acc:.4f}")

print(f"\nTraining complete. Best Val Acc: {best_acc:.4f}")
