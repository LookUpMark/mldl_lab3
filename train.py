import torch
from torch import nn
import os

# Import components from other files
from models.customnet import CustomNet
from dataset.data_loader import get_dataloaders
from eval import validate

# Training function (from MLDL_Lab02.ipynb, modified for device)
def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train() # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device) # Move data to the device

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0: # Print every 100 batches
            print(f'Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)}] \t Loss: {loss.item():.6f} \t Acc: {100. * correct / total:.2f}%')

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'---> End of Epoch {epoch}: \t Loss: {train_loss:.6f} \t Acc: {train_accuracy:.2f}%')

# Main block executed when calling `python train.py`
if __name__ == "__main__":
    
    # 1. Initial Settings
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 128
    dataset_root = 'data/tiny-imagenet-200' # Path to the dataset
    checkpoints_dir = 'checkpoints' # Folder for saved models (checkpoints)

    # Create checkpoints directory if it doesn't exist
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 2. Select Device (GPU if available, else CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # 3. Load Data
    train_loader, val_loader = get_dataloaders(root_dir=dataset_root, batch_size=batch_size)
    
    # Check if dataloaders were loaded correctly
    if train_loader is None or val_loader is None:
        print("Could not load data. Exiting.")
        exit() # Exit script if data not found

    # 4. Initialize Model, Loss, and Optimizer (from MLDL_Lab02.ipynb)
    model = CustomNet().to(device) # Move model to device
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # 5. Training and Validation Loop
    best_acc = 0
    print("Starting training...")

    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer, device)
        val_accuracy = validate(model, val_loader, criterion)

        # Save the best model
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            print(f"New best accuracy: {best_acc:.2f}%. Saving model to {checkpoints_dir}/best_model.pth")
            torch.save(model.state_dict(), f'{checkpoints_dir}/best_model.pth')

    print("Training finished!")
    print(f'Best validation accuracy: {best_acc:.2f}%')