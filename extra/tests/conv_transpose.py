#!/usr/bin/env python
from __future__ import annotations
import unittest
import torch
import random
import psutil
import gc
from time import time

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.neural_net import NN

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)


class ConvAutoencoder(NN):
    """Simple convolutional autoencoder using ConvTranspose2d for the decoder."""

    def __init__(
            self,
            input_shape: tuple[int],
            latent_channels: int=16,
            save: bool=False,
            device: str="cpu"
        ) -> None:
        super().__init__(input_shape, save, device)
        C = input_shape[0]

        # Encoder
        self.conv(in_channels=C, out_channels=latent_channels, kernel_size=3, stride=2, padding=1)
        self.relu()
        # Decoder
        self.conv_transpose(in_channels=latent_channels, out_channels=C, kernel_size=3, stride=2, padding=1, output_padding=1)


class ConvTransposeNet(NN):
    """Network with ConvTranspose2d layers for upsampling."""

    def __init__(
            self,
            input_shape: tuple[int],
            num_classes: int,
            save: bool=False,
            device: str="cpu"
        ) -> None:
        super().__init__(input_shape, save, device)
        C = input_shape[0]

        # Downsample
        self.conv(in_channels=C, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.relu()
        # Upsample with ConvTranspose2d
        self.conv_transpose(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu()
        self.adaptiveavgpool((1,1))
        self.flatten()
        self.linear(in_features=32, out_features=num_classes)


def get_memory_usage() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


class TestConvAutoencoder(unittest.TestCase):
    def test_conv_autoencoder(self) -> None:
        for test_num in range(5):
            print(f"\n--------------------------------")
            print(f"Conv Autoencoder Test {test_num + 1}/5:")

            gc.collect()
            start_mem = get_memory_usage()
            start_time = time()

            w = random.choice([16, 20, 24, 28, 32])
            C = random.choice([1, 3])
            latent_channels = random.choice([8, 16, 32])
            input_shape = (C, w, w)
            x = torch.rand(input_shape)

            model = ConvAutoencoder(input_shape=input_shape, latent_channels=latent_channels).to(DEVICE)
            model.eval()
            forward_pass = model(x)

            batch_size = random.randint(1, 8)
            matrix_computer = KnowledgeMatrixComputer(model, batch_size=batch_size)
            mat = matrix_computer.forward(x)
            diff = torch.norm(forward_pass.flatten() - mat.sum(1)).item()

            end_time = time()
            end_mem = get_memory_usage()

            self.assertAlmostEqual(diff, 0, delta=0.1, msg=f"rep and forward_pass differ by {diff}.")

            print(f"  Input: {input_shape}  Latent Ch: {latent_channels}  Batch: {batch_size}")
            print(f"  Diff: {diff:.6e}  Time: {end_time-start_time:.4f}s  Mem: {end_mem-start_mem:.2f}MB")


class TestConvTransposeNet(unittest.TestCase):
    def test_conv_transpose_classifier(self) -> None:
        for test_num in range(5):
            print(f"\n--------------------------------")
            print(f"ConvTranspose Classifier Test {test_num + 1}/5:")

            gc.collect()
            start_mem = get_memory_usage()
            start_time = time()

            w = random.choice([16, 20, 24, 28, 32])
            C = random.choice([1, 3])
            num_classes = random.randint(2, 20)
            input_shape = (C, w, w)
            x = torch.rand(input_shape)

            model = ConvTransposeNet(input_shape=input_shape, num_classes=num_classes).to(DEVICE)
            model.eval()
            forward_pass = model(x)

            batch_size = random.randint(1, 8)
            matrix_computer = KnowledgeMatrixComputer(model, batch_size=batch_size)
            mat = matrix_computer.forward(x)
            diff = torch.norm(forward_pass - mat.sum(1)).item()

            end_time = time()
            end_mem = get_memory_usage()

            self.assertAlmostEqual(diff, 0, delta=0.1, msg=f"rep and forward_pass differ by {diff}.")

            print(f"  Input: {input_shape}  Classes: {num_classes}  Batch: {batch_size}")
            print(f"  Diff: {diff:.6e}  Time: {end_time-start_time:.4f}s  Mem: {end_mem-start_mem:.2f}MB")


if __name__ == "__main__":
    unittest.main()
