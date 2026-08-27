import os
from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

EPOCHS: int = 90

"""AlexNet uses ReLU (Rectified Linear Unit) as the activation function"""
def ReLU(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp_(x, min=0) # Fixes the minimum value to 0

class AlexNet(nn.Module):
    """AlexNet model architecture from scratch.
    
    It contains 8 layers: 5 convolutional layers and 3 max pooling layers, followed by 3 fully connected layers. 
    The network was originally written in CUDA, C++, but has been reimplemented in PyTorch before. 

    Layer Breakdown:

    1. Input: 3x224x224 pixel image with 3 color channels (RGB)
    2. C1 (Convolution): 11x11 filters, 96 feature maps, stride 4, output size 54x54x96
    3. S2 (MaxPool): 3x3 window, stride 2, output size 26x26x96 # MaxPool, unlike AvgPool extracts just the maximum value from a region, not the mean
    4. C3 (Convolution): 5x5 filters, 256 feature maps, pad 2, output size 26x26x256
    5. S4 (MaxPool): 3x3 window, stride 2, output size 12x12x256
    6. C5 (Convolution): 3x3 filters, 384 feature maps, pad 1, output size 12x12x384
    7. C6 (Convolution): 3x3 filters, 384 feature maps, pad 1, output size 12x12x384
    8. C7 (Convolution): 3x3 filters, 256 feature maps, pad 1, output size 12x12x256
    9. S8 (MaxPool): 3x3 filters, stride 2, output size 5x5x256
    10. F9 (Fully Connected Layer): 4096 neurons (will implement nn.Flatten -> 5x5x256 = 6400 connected to 4096)
    11. D9 (Dropout): p = 0.5 (Deactivates randomly 50% of F9 connections)
    12. F10 (Fully Connected Layer): 4096 neurons
    13. D10 (Dropout): p = 0.5 (Deactivates randomly 50% of F10 connections)
    14. F11 (Fully Connected Layer - Output): 1000 neurons (the classes from the ImageNet competition)


    Notes taken while writing this Layer Breakdown:

    - First MaxPool output shape is calculated: H_out = floor((H_in - Window) / Stride) + 1, in our case H_out = floor((54 - 3) / 2) + 1 = 25 + 1 = 26 -> Layer output: 26x26x96
    - At the near final layers, multiple convolutional layers are stacked, because they retain more complex information (for eg. one 7x7 convolution has one ReLU at the end, but three 3x3 convolutions have three intercalated ReLUs, which enables the network to be more capable of learning complex and subtle mathematical decisions, becoming a more powerful mathematical function at a high-level)
    - At the near final layers (5 and 6), even though the number of feature maps remains the same, the network is retaining different information due to the fact the input of a layer is the output of the previous one, hence these are just further scaling the number of parameters the network has. On a more abstract note, let's say the fifth layer detects simple geometrical structures, and the sixth layer uses those to detect more complex features using those structures.
    - A modification in the padding between 2 consecutive convolutional layers is a geometric trick. A larger padding retains the rezolution unchanged to enable the network to process the information again at the same level of detail. A smaller padding (~0) enables the image to become smaller, due to the hidden layer transformations. 
    - F10 basically understands F9's output and tries to find high-level correlations (Link this to kind of a probability calculation, based on the F9 output, F9 translates the geometry into real objects and F10 basically calculates how likely of each 1000 labels that particular image is, probabilistically)
    - The number 4096 was chosen empirically by the AlexNet authors, they have tested 1024, 2048, and 4096. And the latter has proven to give the highest accuracy in the competition 
    """
    def __init__(self) -> None:
        super().__init__()
        # in_channels=3 because the img. now is in RGB format, unlike the previous LeNet5
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.d6 = nn.Dropout(p=0.5)
        self.f6 = nn.Linear(in_features=256*5*5, out_features=4096) # in_features[C, H, W] = [256, 5, 5] -> 6400
        
        self.d7 = nn.Dropout(p=0.5)
        self.f7 = nn.Linear(in_features=4096, out_features=4096)
        
        self.f8 = nn.Linear(in_features=4096, out_features=1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ReLU(self.conv1(x))
        x = self.maxpool1(x)

        x = ReLU(self.conv2(x))
        x = self.maxpool2(x)

        x = ReLU(self.conv3(x))
        x = ReLU(self.conv4(x))
        x = ReLU(self.conv5(x))
        x = self.maxpool5(x)

        # Flatten function
        # start_dim=1 assures that we flatten just [C, H, W], without the batch
        x = torch.flatten(x, start_dim=1)

        x = self.d6(x)
        x = ReLU(self.f6(x))

        x = self.d7(x)
        x = ReLU(self.f7(x))

        # Final output layer (w/o ReLU in order to keep the logits)
        x = self.f8(x)

        return x
    
    # Now the fit function as also implemented in the former LeNet-5 project
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, device: Optional[Union[torch.device, str]] = None, epochs=EPOCHS):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)

        self.to(device) # Moves the model weights to the device

        # Exact optimizer configuration AlexNet used
        optimizer = torch.optim.SGD(self.parameters(), momentum=0.9, weight_decay=0.0005, lr=0.01)
        # The authors implemented learning rate scheduling by observing how the loss validation loss behaved using checkpoints during training. For the sake of simplicity (not checkpointing), we will use the similar PyTorch ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5, # Number of epochs to wait for improvement before dropping
        )
        # Unlike LeNet-5, AlexNet used the basic, standard, CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            self.train() 
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                outputs: torch.Tensor = self(inputs)
                loss: torch.Tensor = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if batch_idx % 100 == 0:
                    print(f"Batch: {batch_idx:03d} | Batch Loss: {loss.item():.4f}")

            epoch_loss: float = running_loss / len(train_loader)

            val_accuracy, val_loss = self.evaluate(val_loader, criterion, device=device)
            scheduler.step(val_loss)

            print(f"Epoch {epoch} completed | Average Loss: {epoch_loss:.4f}")

    def evaluate(self, criterion: Optional[nn.modules.loss._Loss] = None, val_loader: DataLoader, device: Optional[Union[torch.device, str]], verbose: bool = False) -> Tuple[float, float]:
        """Evaluates the model on the validation dataset and returns the accuracy."""
        if device is None:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)

        self.to(device) # Moves the model weights to the device
        self.eval() # Basically turns off dropout layers and freezes batch normalization updates

        correct = 0
        total = 0
        running_val_loss: float = 0.0
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs: torch.Tensor = self(images)

                loss: torch.Tensor = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                # This time, as for predicting the labels, unlike the LeNet-5 model, we will use the torch.argmax function, the standard for modern Deep Neural Networks
                predicted: torch.Tensor = torch.argmax(outputs, dim=1) # dim 1 represents the logits, while dim 0 represents the outputs

                total += labels.size(0) # the .size(0) method (equal to .shape[0]) returns the length of the first tensor dimension, the exact number of images from the current batch
                correct += (predicted == labels).sum().item() # .sum() basically computes the exact number of correct predictions from the batch; .item() extracts the pure numerical value from the tensor

            accuracy: float = 100 * correct / total
            avg_val_loss: float = running_val_loss / len(val_loader)

            if verbose:
                print(f"Total samples evaluated: {total}")
                print(f"Correct predictions: {correct}")

            print(f"Total accuracy on the validation split: {accuracy:.2f}")

            return accuracy, avg_val_loss

    # Saving and loading the model's state dictionary to/from a file for later use.
    # The default path sits next to this file, so it does not depend on the directory
    # the script or notebook happens to be run from.
    DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alexnet_model.pth')

    def load(self, path: Optional[str]=None, device: Optional[Union[torch.device, str]] = None) -> None:
         """Loads the model's state dictionary from a file."""
         path: str = path or self.DEFAULT_WEIGHTS
         
         if device is None:
              device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         else:
              device = torch.device(device)

         # Load the weights on the specified device
         state_dict = torch.load(path, map_location=device, weights_only=True)
         self.load_state_dict(state_dict)
         self.to(device)

         print(f"AlexNet model loaded from {path} to {device}")

    def save(self, path: Optional[str] = None) -> None:
         """Saves the model's state dictionary to a file."""
         path: str = path or self.DEFAULT_WEIGHTS
         
         # Ensure the directory exists before saving
         dir_name: str = os.path.dirname(path)
         if dir_name:
              os.makedirs(dir_name, exist_ok=True)

         torch.save(self.state_dict(), path)
         print(f"AlexNet model saved to {path}")