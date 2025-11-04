import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

# Define the transforms (from MLDL_Lab02.ipynb)
transform = T.Compose([
    T.Resize((224, 224)),  # Resize to fit the network's input dimensions
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_dataloaders(root_dir='data/tiny-imagenet-200', batch_size=128):
    """
    Creates and returns the training and validation DataLoaders.
    'root_dir' is the path to the downloaded 'tiny-imagenet-200' folder.
    """
    train_path = f'{root_dir}/train'
    val_path = f'{root_dir}/val'

    # Create Datasets using ImageFolder
    try:
        tiny_imagenet_dataset_train = ImageFolder(root=train_path, transform=transform)
        tiny_imagenet_dataset_val = ImageFolder(root=val_path, transform=transform)
    except FileNotFoundError:
        print(f"Error: Dataset directory not found in {root_dir}")
        print("Make sure you have downloaded and prepared the dataset before running the script.")
        return None, None

    print(f"Found training dataset: {len(tiny_imagenet_dataset_train)} samples.")
    print(f"Found validation dataset: {len(tiny_imagenet_dataset_val)} samples.")
    print(f"Number of classes: {len(tiny_imagenet_dataset_train.classes)}")

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        tiny_imagenet_dataset_train, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        tiny_imagenet_dataset_val, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader