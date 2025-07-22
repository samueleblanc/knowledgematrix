"""
    Example on how to use NN and KnowledgeMatrixComputer.
"""
import torch

from knowledgematrix.neural_net import NN
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

from knowledgematrix.models.alexnet import AlexNet  # Import an existing NN


class SimpleMLP(NN):  # Build your own NN

    def __init__(
            self, 
            input_shape: tuple[int],
            num_classes: int,
            save: bool=False, 
            device: str="cpu"
        ) -> None:
        super().__init__(input_shape, save, device)

        self.flatten()
        self.linear(in_features=self.get_input_size(), out_features=256)
        self.relu()
        self.linear(in_features=256, out_features=512)
        self.relu()
        self.linear(in_features=512, out_features=512)
        self.relu()
        self.linear(in_features=512, out_features=num_classes)


def example_mlp():
    print("Simple MLP")
    # Get the MLP you just implemented
    mlp = SimpleMLP(
        input_shape=(1,28,28),
        num_classes=10
    )
    # Get the method to compute the knowledge matrices associated to the MLP
    matrix_computer = KnowledgeMatrixComputer(mlp, batch_size=32)

    x = torch.randn(mlp.input_shape)  # Input
    out = mlp.forward(x)  # Output
    mat = matrix_computer.forward(x)  # Knowledge matrix
    print(out.shape)
    print(mat.shape)

    print(f"Difference = {torch.linalg.norm(out - mat.sum(1))}\n")  # Should be close to zero!


def example_alexnet():
    print("AlexNet")
    # Get AlexNet, which was already implemented
    alexnet = AlexNet(
        input_shape=(3,63,63),
        num_classes=10
    )
    # Get the method to compute the knowledge matrices associated to the AlexNet
    matrix_computer = KnowledgeMatrixComputer(alexnet, batch_size=8)

    x = torch.randn(alexnet.input_shape)  # Input
    out = alexnet.forward(x)  # Output
    mat = matrix_computer.forward(x)  # Knowledge matrix
    print(out.shape)
    print(mat.shape)

    print(f"Difference = {torch.linalg.norm(out - mat.sum(1))}\n")  # Should be close to zero!


def main():
    example_mlp()
    example_alexnet()


if __name__ == "__main__":
    main()