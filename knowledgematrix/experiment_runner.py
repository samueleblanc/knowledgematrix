import os
import logging
from typing import Iterable

import torch

from knowledgematrix.neural_net import NN
from knowledgematrix.dataset_computer import DatasetComputer

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
        Orchestrates knowledge matrix computation across multiple weight checkpoints.

        Manages the experiment directory layout, loading weight snapshots, computing
        matrices for each sample in a dataset, and compressing results. Designed for
        SLURM cluster workflows where results are computed on a compute node and
        transferred to a login node.

        The data argument to run() and run_single() must support re-iteration
        (e.g., Dataset, Subset, list) when processing multiple weight checkpoints.
        Single-use generators will only work with run_single().

        Args:
            model (NN): The model architecture. Weights will be swapped via load_state_dict.
            experiment_dir (str): Path to experiments/<experiment-name>/. Expects weights
                in experiment_dir/weights/ and saves matrices to experiment_dir/matrices/.
            batch_size (int): Number of knowledge matrix columns processed at once.
            device (str | None): The device to use for computation. If None, uses the model's device.
    """

    def __init__(
            self,
            model: NN,
            experiment_dir: str,
            batch_size: int = 1,
            device: str | None = None
    ) -> None:
        self.model = model
        self.experiment_dir = experiment_dir
        self.batch_size = batch_size
        self.device = device
        self._computer = DatasetComputer(model, batch_size=batch_size, device=device)
        os.makedirs(os.path.join(experiment_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "matrices"), exist_ok=True)

    def get_weight_paths(self) -> list[str]:
        """
            List and sort all .pt files in experiment_dir/weights/.

            Returns:
                list[str]: Sorted list of full paths to weight checkpoint files.
        """
        weights_dir = os.path.join(self.experiment_dir, "weights")
        files = [
            os.path.join(weights_dir, f)
            for f in sorted(os.listdir(weights_dir))
            if f.endswith(".pt")
        ]
        return files

    def run(
            self,
            data: Iterable,
            weight_paths: list[str] | None = None,
            resume: bool = True
    ) -> None:
        """
            Run knowledge matrix computation for each weight checkpoint.

            For each weight checkpoint: loads the state_dict, sets the model to eval mode,
            computes matrices for all samples, and compresses the results.

            Args:
                data: Any iterable yielding tensors or (tensor, label) tuples.
                    Must support re-iteration when processing multiple checkpoints.
                weight_paths (list[str] | None): List of .pt file paths. If None, uses
                    all weights from get_weight_paths().
                resume (bool): If True, skips checkpoints that already have matrices.tar.gz,
                    and skips individual samples that are already computed.
        """
        if weight_paths is None:
            weight_paths = self.get_weight_paths()

        for weight_path in weight_paths:
            weight_name = os.path.splitext(os.path.basename(weight_path))[0]
            output_dir = os.path.join(self.experiment_dir, "matrices", weight_name)
            archive_path = os.path.join(output_dir, "matrices.tar.gz")

            if resume and os.path.exists(archive_path):
                logger.info(f"Skipping checkpoint {weight_name} (archive already exists)")
                continue

            logger.info(f"Processing checkpoint: {weight_name}")
            state_dict = torch.load(weight_path, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            self._computer.compute(data, output_dir, resume=resume)
            self._computer.compress(output_dir)
            logger.info(f"Completed checkpoint: {weight_name}")

    def run_single(
            self,
            data: Iterable,
            weight_path: str,
            resume: bool = True
    ) -> None:
        """
            Run knowledge matrix computation for a single weight checkpoint.

            Args:
                data: Any iterable yielding tensors or (tensor, label) tuples.
                weight_path (str): Path to a single .pt weight checkpoint file.
                resume (bool): If True, skips samples that are already computed.
        """
        self.run(data, weight_paths=[weight_path], resume=resume)
