"""
    Implementation of ResNet-18
"""
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

from knowledgematrix.neural_net import NN


class ResNet18(NN):
    """
        The ResNet-18 model.

        Args:
            input_shape (Tuple[int]): The shape of the input to the network.
            num_classes (int): The number of classes in the dataset.
            save (bool): Whether to save the activations and preactivations of the network.
            pretrained (bool): Whether to use pretrained weights.
            device (str): The device to run the network on.
    """
    def __init__(
            self, 
            input_shape: tuple[int],
            num_classes: int,
            save: bool=False, 
            pretrained: bool=False,
            device: str="cpu"
        ) -> None:
        super().__init__(input_shape, save, device)
    
        if pretrained:
            if input_shape[0] != 3 or num_classes != 1000:
                raise ValueError("ResNet18 was trained on images with 3 channels and 1000 classes. Please use input_shape=(3, -, -) and num_classes=1000 for pretrained ResNet18.")
            for layer in resnet18(weights=ResNet18_Weights.DEFAULT).children():
                if isinstance(layer, nn.Sequential):
                    for basic_block in layer.children():
                        start_skip = self.get_num_layers()
                        for sublayer in basic_block.children():
                            if isinstance(sublayer, (nn.Conv2d, nn.BatchNorm2d)):
                                self.layers.append(sublayer)
                            elif isinstance(sublayer, nn.ReLU):
                                self.relu()
                        end_skip = self.get_num_layers()
                        self.residual(start_skip, end_skip)
                        for sublayer in basic_block.children():
                            if isinstance(sublayer, nn.Sequential):
                                for i, downsample_layer in enumerate(sublayer.children()):
                                    self.residuals[end_skip][0][1][i] = downsample_layer
                elif isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    self.layers.append(layer)
                elif isinstance(layer, nn.ReLU):
                    self.relu()
                elif isinstance(layer, nn.AdaptiveAvgPool2d):
                    self.adaptiveavgpool(output_size=layer.output_size)
                    self.flatten()
                elif isinstance(layer, nn.MaxPool2d):
                    self.maxpool(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
        else:
            self.conv(self.input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.batchnorm(64)
            self.relu()
            self.maxpool(kernel_size=3, stride=2, padding=1)

            # First block
            start_skip = self.get_num_layers()
            self.conv(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.batchnorm(64)
            self.relu()
            self.conv(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.batchnorm(64)
            end_skip = self.get_num_layers()
            self.residual(start_skip, end_skip)

            start_skip = self.get_num_layers()
            self.conv(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.batchnorm(64)
            self.relu()
            self.conv(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.batchnorm(64)
            end_skip = self.get_num_layers()
            self.residual(start_skip, end_skip)

            # Second block
            start_skip = self.get_num_layers()
            self.conv(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
            self.batchnorm(128)
            self.relu()
            self.conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
            self.batchnorm(128)
            end_skip = self.get_num_layers()
            self.residual(start_skip, end_skip)

            start_skip = self.get_num_layers()
            self.conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
            self.batchnorm(128)
            self.relu()
            self.conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
            self.batchnorm(128)
            end_skip = self.get_num_layers()
            self.residual(start_skip, end_skip)

            # Third block
            start_skip = self.get_num_layers()
            self.conv(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
            self.batchnorm(256)
            self.relu()
            self.conv(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
            self.batchnorm(256)
            end_skip = self.get_num_layers()
            self.residual(start_skip, end_skip)

            start_skip = self.get_num_layers()
            self.conv(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
            self.batchnorm(256)
            self.relu()
            self.conv(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
            self.batchnorm(256)
            end_skip = self.get_num_layers()
            self.residual(start_skip, end_skip)

            # Forth block
            start_skip = self.get_num_layers()
            self.conv(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
            self.batchnorm(512)
            self.relu()
            self.conv(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
            self.batchnorm(512)
            end_skip = self.get_num_layers()
            self.residual(start_skip, end_skip)

            start_skip = self.get_num_layers()
            self.conv(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
            self.batchnorm(512)
            self.relu()
            self.conv(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
            self.batchnorm(512)
            end_skip = self.get_num_layers()
            self.residual(start_skip, end_skip)

            self.adaptiveavgpool((1,1))
            self.flatten()
            self.linear(in_features=512, out_features=num_classes)
