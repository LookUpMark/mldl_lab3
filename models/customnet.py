from torch import nn

# Defines the custom neural network
# Taken from MLDL_Lab02.ipynb
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Layer definitions
        
        # Conv layer 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv layer 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the input size for the linear layer
        # Input image size is 224x224
        # After conv1 and pool1: 224/2 = 112
        # After conv2 and pool2: 112/2 = 56
        # Channels after conv2: 128
        # Input features = 128 * 56 * 56
        self.fc1 = nn.Linear(128 * 56 * 56, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass
        # B x 3 x 224 x 224
        x = self.conv1(x).relu() # B x 64 x 224 x 224
        x = self.pool1(x) # B x 64 x 112 x 112

        x = self.conv2(x).relu() # B x 128 x 112 x 112
        x = self.pool2(x) # B x 128 x 56 x 56

        x = x.flatten(start_dim=1) # B x (128 * 56 * 56)

        x = self.fc1(x)

        return x