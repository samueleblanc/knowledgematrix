#!/usr/bin/env python
import unittest
import os
import tempfile
import shutil
import tarfile

import torch
from torch.utils.data import TensorDataset, Subset

from knowledgematrix.neural_net import NN
from knowledgematrix.dataset_computer import DatasetComputer
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)


class SmallMLP(NN):

    def __init__(
            self,
            input_shape: tuple[int],
            num_classes: int,
            save: bool = False,
            device: str = "cpu"
    ) -> None:
        super().__init__(input_shape, save, device)
        self.flatten()
        self.linear(in_features=self.get_input_size(), out_features=32)
        self.relu()
        self.linear(in_features=32, out_features=num_classes)


class TestDatasetComputer(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.input_shape = (1, 4, 4)
        self.num_classes = 3
        self.model = SmallMLP(self.input_shape, self.num_classes).to(DEVICE)
        self.model.eval()
        self.data = [torch.randn(self.input_shape) for _ in range(5)]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_compute_correctness(self) -> None:
        """
            Test that each saved knowledge matrix satisfies mat.sum(1) ≈ model(x).
        """
        output_dir = os.path.join(self.tmp_dir, "matrices")
        computer = DatasetComputer(self.model, batch_size=16, device=DEVICE)
        computer.compute(self.data, output_dir)

        for i, x in enumerate(self.data):
            filepath = os.path.join(output_dir, f"sample_{i}.pt")
            self.assertTrue(os.path.exists(filepath), f"sample_{i}.pt not found")

            mat = torch.load(filepath, weights_only=True)
            forward_pass = self.model(x)
            diff = torch.norm(forward_pass - mat.sum(1)).item()

            self.assertAlmostEqual(
                first=diff,
                second=0,
                places=None,
                msg=f"Sample {i}: mat.sum(1) and forward_pass differ by {diff}.",
                delta=0.1
            )

    def test_compute_resume(self) -> None:
        """
            Test that resume skips already-computed samples.
        """
        output_dir = os.path.join(self.tmp_dir, "matrices")
        computer = DatasetComputer(self.model, batch_size=16, device=DEVICE)

        # Compute first 3 samples
        computer.compute(self.data[:3], output_dir)
        mtimes = {}
        for i in range(3):
            filepath = os.path.join(output_dir, f"sample_{i}.pt")
            mtimes[i] = os.path.getmtime(filepath)

        # Compute all 5 with resume — first 3 should be skipped
        computer.compute(self.data, output_dir, resume=True)

        for i in range(3):
            filepath = os.path.join(output_dir, f"sample_{i}.pt")
            self.assertEqual(
                os.path.getmtime(filepath),
                mtimes[i],
                f"sample_{i}.pt was re-computed despite resume=True"
            )

        for i in range(5):
            filepath = os.path.join(output_dir, f"sample_{i}.pt")
            self.assertTrue(os.path.exists(filepath), f"sample_{i}.pt not found after resume")

    def test_compress(self) -> None:
        """
            Test that compress creates a valid .tar.gz with correct entries.
        """
        output_dir = os.path.join(self.tmp_dir, "matrices")
        computer = DatasetComputer(self.model, batch_size=16, device=DEVICE)
        computer.compute(self.data, output_dir)

        archive_path = computer.compress(output_dir)
        self.assertTrue(os.path.exists(archive_path), "Archive not created")
        self.assertTrue(archive_path.endswith(".tar.gz"), "Archive has wrong extension")

        with tarfile.open(archive_path, "r:gz") as tar:
            names = sorted(tar.getnames())
            expected = sorted([f"sample_{i}.pt" for i in range(5)])
            self.assertEqual(names, expected, f"Archive contents {names} != expected {expected}")

    def test_subset(self) -> None:
        """
            Test that DatasetComputer works with torch.utils.data.Subset (tuple extraction).
        """
        output_dir = os.path.join(self.tmp_dir, "matrices")
        tensors = torch.stack(self.data)
        labels = torch.zeros(5, dtype=torch.long)
        dataset = TensorDataset(tensors, labels)
        subset = Subset(dataset, [0, 2, 4])

        computer = DatasetComputer(self.model, batch_size=16, device=DEVICE)
        computer.compute(subset, output_dir)

        # Should have 3 files (indices 0, 1, 2 in the subset iteration)
        for i in range(3):
            filepath = os.path.join(output_dir, f"sample_{i}.pt")
            self.assertTrue(os.path.exists(filepath), f"sample_{i}.pt not found")

        # Verify correctness for each subset element
        subset_indices = [0, 2, 4]
        for i, idx in enumerate(subset_indices):
            mat = torch.load(os.path.join(output_dir, f"sample_{i}.pt"), weights_only=True)
            forward_pass = self.model(self.data[idx])
            diff = torch.norm(forward_pass - mat.sum(1)).item()

            self.assertAlmostEqual(
                first=diff,
                second=0,
                places=None,
                msg=f"Subset sample {i} (original idx {idx}): differ by {diff}.",
                delta=0.1
            )

    def test_file_naming(self) -> None:
        """
            Test that output files are named sample_0.pt, sample_1.pt, etc.
        """
        output_dir = os.path.join(self.tmp_dir, "matrices")
        computer = DatasetComputer(self.model, batch_size=16, device=DEVICE)
        computer.compute(self.data, output_dir)

        files = sorted(os.listdir(output_dir))
        expected = [f"sample_{i}.pt" for i in range(5)]
        self.assertEqual(files, expected, f"File names {files} != expected {expected}")


if __name__ == "__main__":
    unittest.main()
