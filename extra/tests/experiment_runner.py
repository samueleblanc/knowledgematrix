#!/usr/bin/env python
import unittest
import os
import tempfile
import shutil
import tarfile

import torch

from knowledgematrix.neural_net import NN
from knowledgematrix.experiment_runner import ExperimentRunner

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


class TestExperimentRunner(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.input_shape = (1, 4, 4)
        self.num_classes = 3
        self.model = SmallMLP(self.input_shape, self.num_classes).to(DEVICE)
        self.data = [torch.randn(self.input_shape) for _ in range(5)]
        self.weight_names = ["epoch_0", "epoch_10", "epoch_50"]

        # Save weight checkpoints with randomized weights
        weights_dir = os.path.join(self.tmp_dir, "weights")
        os.makedirs(weights_dir)
        for name in self.weight_names:
            for param in self.model.parameters():
                param.data = torch.randn_like(param.data)
            torch.save(
                self.model.state_dict(),
                os.path.join(weights_dir, f"{name}.pt")
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_get_weight_paths(self) -> None:
        """
            Test that get_weight_paths returns sorted full paths.
        """
        runner = ExperimentRunner(self.model, self.tmp_dir, batch_size=16, device=DEVICE)
        paths = runner.get_weight_paths()

        self.assertEqual(len(paths), 3)
        for path in paths:
            self.assertTrue(os.path.exists(path), f"{path} does not exist")
            self.assertTrue(path.endswith(".pt"))

        # Verify sorted order
        basenames = [os.path.basename(p) for p in paths]
        self.assertEqual(basenames, sorted(basenames))

    def test_run(self) -> None:
        """
            Test run() computes matrices for all checkpoints with correct layout.
        """
        runner = ExperimentRunner(self.model, self.tmp_dir, batch_size=16, device=DEVICE)
        runner.run(self.data)

        for weight_name in self.weight_names:
            matrix_dir = os.path.join(self.tmp_dir, "matrices", weight_name)
            self.assertTrue(os.path.isdir(matrix_dir), f"Directory {matrix_dir} not created")

            archive_path = os.path.join(matrix_dir, "matrices.tar.gz")
            self.assertTrue(os.path.exists(archive_path), f"Archive not created for {weight_name}")

            # Verify archive contents
            with tarfile.open(archive_path, "r:gz") as tar:
                names = sorted(tar.getnames())
                expected = sorted([f"sample_{i}.pt" for i in range(5)])
                self.assertEqual(names, expected)

        # Verify matrix correctness for each checkpoint
        for weight_name in self.weight_names:
            weight_path = os.path.join(self.tmp_dir, "weights", f"{weight_name}.pt")
            state_dict = torch.load(weight_path, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            matrix_dir = os.path.join(self.tmp_dir, "matrices", weight_name)
            for i, x in enumerate(self.data):
                filepath = os.path.join(matrix_dir, f"sample_{i}.pt")
                mat = torch.load(filepath, weights_only=True)
                forward_pass = self.model(x)
                diff = torch.norm(forward_pass - mat.sum(1)).item()

                self.assertAlmostEqual(
                    first=diff,
                    second=0,
                    places=None,
                    msg=f"Checkpoint {weight_name}, sample {i}: differ by {diff}.",
                    delta=0.1
                )

    def test_run_resume(self) -> None:
        """
            Test that resume skips checkpoints with existing archives.
        """
        runner = ExperimentRunner(self.model, self.tmp_dir, batch_size=16, device=DEVICE)
        runner.run(self.data)

        # Record archive modification times
        mtimes = {}
        for weight_name in self.weight_names:
            archive_path = os.path.join(self.tmp_dir, "matrices", weight_name, "matrices.tar.gz")
            mtimes[weight_name] = os.path.getmtime(archive_path)

        # Run again with resume
        runner.run(self.data, resume=True)

        # Verify archives were not re-created
        for weight_name in self.weight_names:
            archive_path = os.path.join(self.tmp_dir, "matrices", weight_name, "matrices.tar.gz")
            self.assertEqual(
                os.path.getmtime(archive_path),
                mtimes[weight_name],
                f"Archive for {weight_name} was re-created despite resume=True"
            )

    def test_run_single(self) -> None:
        """
            Test run_single() processes only the specified checkpoint.
        """
        runner = ExperimentRunner(self.model, self.tmp_dir, batch_size=16, device=DEVICE)
        weight_path = os.path.join(self.tmp_dir, "weights", "epoch_10.pt")
        runner.run_single(self.data, weight_path)

        # Only epoch_10 should have matrices
        self.assertTrue(
            os.path.isdir(os.path.join(self.tmp_dir, "matrices", "epoch_10")),
            "epoch_10 matrices directory not created"
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.tmp_dir, "matrices", "epoch_10", "matrices.tar.gz")),
            "epoch_10 archive not created"
        )

        # Other checkpoints should NOT have matrices
        for name in ["epoch_0", "epoch_50"]:
            self.assertFalse(
                os.path.exists(os.path.join(self.tmp_dir, "matrices", name, "matrices.tar.gz")),
                f"{name} should not have been processed"
            )


if __name__ == "__main__":
    unittest.main()
