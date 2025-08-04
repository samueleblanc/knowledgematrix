"""
    Implementation of ResNet-18
"""
from knowledgematrix.neural_net import NN


class ResNet18(NN):
    """
        The ResNet-18 model.

        Args:
            input_shape (Tuple[int]): The shape of the input to the network.
            num_classes (int): The number of classes in the dataset.
            save (bool): Whether to save the activations and preactivations of the network.
            device (str): The device to run the network on.
    """
    def __init__(
            self, 
            input_shape: tuple[int],
            num_classes: int,
            save: bool=False, 
            device: str="cpu"
        ) -> None:
        super().__init__(input_shape, save, device)
    
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
        self.relu()

        start_skip = self.get_num_layers()
        self.conv(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm(64)
        self.relu()
        self.conv(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm(64)
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        self.relu()

        # Second block
        start_skip = self.get_num_layers()
        self.conv(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchnorm(128)
        self.relu()
        self.conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm(128)
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        self.relu()

        start_skip = self.get_num_layers()
        self.conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm(128)
        self.relu()
        self.conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm(128)
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        self.relu()

        # Third block
        start_skip = self.get_num_layers()
        self.conv(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchnorm(256)
        self.relu()
        self.conv(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm(256)
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        self.relu()

        start_skip = self.get_num_layers()
        self.conv(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm(256)
        self.relu()
        self.conv(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm(256)
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        self.relu()

        # Forth block
        start_skip = self.get_num_layers()
        self.conv(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchnorm(512)
        self.relu()
        self.conv(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm(512)
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        self.relu()

        start_skip = self.get_num_layers()
        self.conv(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm(512)
        self.relu()
        self.conv(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm(512)
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        self.relu()

        self.adaptiveavgpool((1,1))
        self.flatten()
        self.linear(in_features=512, out_features=num_classes)