"""
    Implementation of VGG-11
"""
from torch import nn
from torchvision.models import vgg11, VGG11_Weights
from typing import Union

from knowledgematrix.neural_net import NN


class VGG11(NN):
    """
        The VGG-11 model.

        Args:
            input_shape (Tuple[int]): The shape of the input to the network.
            num_classes (int): The number of classes in the dataset.
            save (bool): Whether to save the activations and preactivations of the network.
            pretrained (bool): Whether to use pretrained weights.
            pretrained_model (Union[vgg11, None]): You can pass in a custom vgg11 model to use as pretrained weights. If None, the default pretrained weights will be used.
            device (str): The device to run the network on.
    """
    def __init__(
            self, 
            input_shape: tuple[int],
            num_classes: int,
            save: bool=False, 
            pretrained: bool=False,
            pretrained_model: Union[vgg11, None]=None,
            device: str="cpu"
        ) -> None:
        super().__init__(input_shape, save, device)

        if pretrained:
            if pretrained_model is None:
                pretrained_model = vgg11(weights=VGG11_Weights.DEFAULT)
            else:
                if not isinstance(pretrained_model, vgg11):
                    raise ValueError("pretrained_model must be an instance of vgg11.")
            for layer in pretrained_model.children():
                if isinstance(layer, nn.Sequential):
                    for sublayer in layer.children():
                        if isinstance(sublayer, (nn.Conv2d, nn.Linear)):
                            self.layers.append(sublayer)
                        elif isinstance(sublayer, nn.ReLU):
                            self.relu()
                        elif isinstance(sublayer, nn.MaxPool2d):
                            self.maxpool(kernel_size=sublayer.kernel_size, stride=sublayer.stride, padding=sublayer.padding)
                        elif isinstance(sublayer, nn.Dropout):
                            self.dropout(p=sublayer.p)
                elif isinstance(layer, nn.AdaptiveAvgPool2d):
                    self.adaptiveavgpool(output_size=layer.output_size)
                    self.flatten()
            if input_shape[0] != 3:
                print(f"Warning: The pretrained model was trained on 3-channel images. The input shape is {input_shape}. The first layer won't have pretrained weights.")
                self.layers[0] = nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1)
            elif num_classes != 1000:
                print(f"Warning: The pretrained model was trained on 1000 classes. The number of classes is {num_classes}. The last layer won't have pretrained weights.")
                self.layers[-1] = nn.Linear(4096, num_classes)
        else:
            # Convolutional Layers
            self.conv(input_shape[0], 64, kernel_size=3, padding=1)
            self.relu()
            self.maxpool(kernel_size=2, stride=2)
            self.conv(64, 128, kernel_size=3, padding=1)
            self.relu()
            self.maxpool(kernel_size=2, stride=2)
            self.conv(128, 256, kernel_size=3, padding=1)
            self.relu()
            self.conv(256, 256, kernel_size=3, padding=1)
            self.relu()
            self.maxpool(kernel_size=2, stride=2)
            self.conv(256, 512, kernel_size=3, padding=1)
            self.relu()
            self.conv(512, 512, kernel_size=3, padding=1)
            self.relu()
            self.maxpool(kernel_size=2, stride=2)
            self.conv(512, 512, kernel_size=3, padding=1)
            self.relu()
            self.conv(512, 512, kernel_size=3, padding=1)
            self.relu()
            self.maxpool(kernel_size=2, stride=2)

            self.flatten()

            # Fully Connected Layers
            self.linear(in_features=512*(self.input_shape[-2]// (2**5))*(self.input_shape[-1] // (2**5)), out_features=4096)
            self.relu()
            self.dropout(0.5)
            self.linear(in_features=4096, out_features=4096)
            self.relu()
            self.dropout(0.5)
            self.linear(in_features=4096, out_features=num_classes)
