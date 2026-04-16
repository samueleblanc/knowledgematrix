import os
import tarfile
import logging
from typing import Iterable

import torch

from knowledgematrix.neural_net import NN
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

logger = logging.getLogger(__name__)


class DatasetComputer:
    """
        Computes knowledge matrices for all samples in a dataset and saves them to disk.

        Wraps KnowledgeMatrixComputer to iterate over a dataset, compute one knowledge matrix
        per sample, and save each as a .pt file. Supports resume (skipping already-computed
        samples) and compression to .tar.gz for cluster workflows.

        Args:
            model (NN): The neural network architecture.
            batch_size (int): Number of knowledge matrix columns processed at once.
                This is NOT a data batch size — it controls how many columns of the
                knowledge matrix are computed simultaneously.
            device (str | None): The device to use for computation. If None, uses the model's device.
    """

    def __init__(
            self,
            model: NN,
            batch_size: int = 1,
            device: str | None = None
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self._computer = KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)

    def compute(
            self,
            data: Iterable,
            output_dir: str,
            resume: bool = True
    ) -> None:
        """
            Compute and save knowledge matrices for each sample in data.

            Iterates over data, computes the knowledge matrix for each sample using
            KnowledgeMatrixComputer, and saves each matrix as sample_{i}.pt.

            Args:
                data: Any iterable yielding tensors or (tensor, label) tuples.
                    Supports Dataset, Subset, lists, and generators.
                output_dir (str): Directory where .pt files will be saved.
                resume (bool): If True, skips samples where sample_{i}.pt already exists.
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, item in enumerate(data):
            filename = f"sample_{i}.pt"
            filepath = os.path.join(output_dir, filename)

            if resume and os.path.exists(filepath):
                logger.info(f"Skipping sample {i} (already exists)")
                continue

            x = item[0] if isinstance(item, (tuple, list)) else item
            mat = self._computer.forward(x)
            torch.save(mat, filepath)
            logger.info(f"Computed sample {i}")

    def compress(
            self,
            output_dir: str,
            archive_path: str | None = None
    ) -> str:
        """
            Compress output_dir contents into a .tar.gz archive.

            Args:
                output_dir (str): Directory containing .pt files to compress.
                archive_path (str | None): Path for the archive. Defaults to
                    {output_dir}/matrices.tar.gz.

            Returns:
                str: The path to the created archive.
        """
        if archive_path is None:
            archive_path = os.path.join(output_dir, "matrices.tar.gz")

        archive_abs = os.path.abspath(archive_path)

        with tarfile.open(archive_path, "w:gz") as tar:
            for entry in sorted(os.listdir(output_dir)):
                entry_path = os.path.join(output_dir, entry)
                if os.path.abspath(entry_path) == archive_abs:
                    continue
                if entry.endswith(".pt"):
                    tar.add(entry_path, arcname=entry)

        logger.info(f"Compressed {output_dir} to {archive_path}")
        return archive_path
