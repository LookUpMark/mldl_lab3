import torch
import wandb # Import wandb

def validate(model, val_loader, criterion, epoch):
    """
    Performs a full validation loop and logs to wandb.
    'epoch' is passed in to log metrics against it.
    """
    model.eval() # Set model to evaluation mode
    val_loss = 0
    correct, total = 0, 0
    
    device = next(model.parameters()).device

    with torch.no_grad(): # Disable gradient calculation
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'\nValidation Loss: {val_loss:.6f} \t Acc: {val_accuracy:.2f}%')
    
    # === WANDB LOGGING (VALIDATION) ===
    wandb.log({
        "epoch": epoch,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })
    
    return val_accuracy