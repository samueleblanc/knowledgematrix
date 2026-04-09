#!/usr/bin/env python
"""
    SLURM cluster test for the knowledgematrix library.

    Runs 4 sequential phases to validate the library on a GPU cluster:
        Phase 1: Smoke test (single KM computation, verify correctness)
        Phase 2: Batch size calibration (find optimal batch_size for H100 80GB)
        Phase 3: DatasetComputer test (20 dummy samples, save/compress)
        Phase 4: ExperimentRunner test (3 fake checkpoints, 10 samples each)

    Usage:
        python3 extra/cluster_test.py
        sbatch extra/cluster_test.sh
"""
import os
import sys
import time
import shutil
import logging
import tempfile

import torch

from knowledgematrix.models.resnet18 import ResNet18
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.dataset_computer import DatasetComputer
from knowledgematrix.experiment_runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)

NUM_CLASSES = 10
CALIBRATION_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
CALIBRATION_INPUT_SIZES = [(3, 32, 32), (3, 224, 224)]
CALIBRATION_WARMUP_RUNS = 1
CALIBRATION_TIMED_RUNS = 3


# ---------------------------------------------------------------------------
# Phase 1: Smoke test
# ---------------------------------------------------------------------------

def phase1_smoke_test() -> bool:
    """Verify a single KM computation reconstructs model output."""
    logger.info("=== Phase 1: Smoke Test ===")

    input_shape = (3, 32, 32)
    model = ResNet18(input_shape=input_shape, num_classes=NUM_CLASSES, device=DEVICE)
    model.eval()

    x = torch.randn(input_shape, device=DEVICE)
    forward_pass = model(x)

    computer = KnowledgeMatrixComputer(model, batch_size=1, device=DEVICE)
    mat = computer.forward(x)

    diff = torch.norm(forward_pass - mat.sum(1)).item()
    passed = diff < 0.1

    logger.info(f"Reconstruction error: {diff:.6e}")
    logger.info(f"Phase 1: {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# Phase 2: Batch size calibration
# ---------------------------------------------------------------------------

def _calibrate_single_input(input_shape: tuple[int, ...]) -> dict:
    """Run calibration for a single input shape. Returns results dict."""
    input_features = 1
    for d in input_shape:
        input_features *= d

    model = ResNet18(input_shape=input_shape, num_classes=NUM_CLASSES, device=DEVICE)
    model.eval()
    x = torch.randn(input_shape, device=DEVICE)

    results = []
    for batch_size in CALIBRATION_BATCH_SIZES:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            computer = KnowledgeMatrixComputer(model, batch_size=batch_size, device=DEVICE)

            # Warmup
            for _ in range(CALIBRATION_WARMUP_RUNS):
                computer.forward(x)

            # Timed runs
            times = []
            for _ in range(CALIBRATION_TIMED_RUNS):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                computer.forward(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)

            avg_time = sum(times) / len(times)
            throughput = input_features / avg_time
            results.append({
                "batch_size": batch_size,
                "avg_time_s": avg_time,
                "throughput_cols_per_s": throughput,
            })
            logger.info(f"  batch_size={batch_size:>5d}  avg_time={avg_time:.4f}s  throughput={throughput:.0f} cols/s")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.info(f"  batch_size={batch_size:>5d}  OOM — stopping calibration")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            raise

    if not results:
        return {"input_shape": input_shape, "results": [], "optimal_batch_size": 1}

    best = min(results, key=lambda r: r["avg_time_s"])
    return {
        "input_shape": input_shape,
        "results": results,
        "optimal_batch_size": best["batch_size"],
    }


def phase2_calibration() -> dict[tuple, int]:
    """Calibrate batch_size for each input shape. Returns {input_shape: optimal_batch_size}."""
    logger.info("=== Phase 2: Batch Size Calibration ===")

    optimal_sizes = {}
    for input_shape in CALIBRATION_INPUT_SIZES:
        logger.info(f"Calibrating for input_shape={input_shape}")
        cal = _calibrate_single_input(input_shape)
        optimal_sizes[input_shape] = cal["optimal_batch_size"]

        # Print table
        logger.info(f"  {'batch_size':>10s}  {'avg_time (s)':>12s}  {'throughput (cols/s)':>20s}")
        logger.info(f"  {'-'*10}  {'-'*12}  {'-'*20}")
        for r in cal["results"]:
            marker = " <-- optimal" if r["batch_size"] == cal["optimal_batch_size"] else ""
            logger.info(
                f"  {r['batch_size']:>10d}  {r['avg_time_s']:>12.4f}  {r['throughput_cols_per_s']:>20.0f}{marker}"
            )
        logger.info(f"  Optimal batch_size for {input_shape}: {cal['optimal_batch_size']}")

    logger.info("Phase 2: DONE")
    return optimal_sizes


# ---------------------------------------------------------------------------
# Phase 3: DatasetComputer test
# ---------------------------------------------------------------------------

def phase3_dataset_computer(batch_size: int) -> bool:
    """Test DatasetComputer with 20 dummy samples."""
    logger.info("=== Phase 3: DatasetComputer Test ===")
    logger.info(f"Using batch_size={batch_size}")

    input_shape = (3, 32, 32)
    model = ResNet18(input_shape=input_shape, num_classes=NUM_CLASSES, device=DEVICE)
    model.eval()

    data = [(torch.randn(input_shape, device=DEVICE), i) for i in range(20)]

    output_dir = tempfile.mkdtemp(prefix="km_dataset_test_")
    try:
        # Compute matrices
        dc = DatasetComputer(model, batch_size=batch_size, device=DEVICE)
        dc.compute(data, output_dir)

        # Verify all 20 files exist
        pt_files = sorted(f for f in os.listdir(output_dir) if f.endswith(".pt"))
        if len(pt_files) != 20:
            logger.error(f"Expected 20 .pt files, found {len(pt_files)}")
            return False

        # Verify reconstruction for each sample
        model.eval()
        max_diff = 0.0
        for i, (x, _) in enumerate(data):
            mat = torch.load(os.path.join(output_dir, f"sample_{i}.pt"), weights_only=True)
            forward_pass = model(x)
            diff = torch.norm(forward_pass - mat.sum(1)).item()
            max_diff = max(max_diff, diff)
            if diff >= 0.1:
                logger.error(f"Sample {i} reconstruction error {diff:.6e} >= 0.1")
                return False

        logger.info(f"All 20 samples verified (max error: {max_diff:.6e})")

        # Test compression
        archive_path = dc.compress(output_dir)
        if not os.path.exists(archive_path):
            logger.error("Compression failed — archive not found")
            return False
        logger.info(f"Archive created: {os.path.getsize(archive_path)} bytes")

        logger.info("Phase 3: PASS")
        return True

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Phase 4: ExperimentRunner test
# ---------------------------------------------------------------------------

def phase4_experiment_runner(batch_size: int) -> bool:
    """Test ExperimentRunner with 3 fake checkpoints and 10 samples."""
    logger.info("=== Phase 4: ExperimentRunner Test ===")
    logger.info(f"Using batch_size={batch_size}")

    input_shape = (3, 32, 32)
    model = ResNet18(input_shape=input_shape, num_classes=NUM_CLASSES, device=DEVICE)
    model.eval()

    experiment_dir = tempfile.mkdtemp(prefix="km_experiment_test_")
    try:
        # Create 3 fake weight checkpoints
        weights_dir = os.path.join(experiment_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        checkpoint_names = ["epoch_0", "epoch_10", "epoch_20"]
        for name in checkpoint_names:
            for param in model.parameters():
                param.data = torch.randn_like(param.data)
            torch.save(model.state_dict(), os.path.join(weights_dir, f"{name}.pt"))

        # Create dummy dataset (10 samples)
        data = [(torch.randn(input_shape, device=DEVICE), i) for i in range(10)]

        # Run experiment
        runner = ExperimentRunner(model, experiment_dir, batch_size=batch_size, device=DEVICE)
        runner.run(data)

        # Verify directory structure
        matrices_dir = os.path.join(experiment_dir, "matrices")
        for name in checkpoint_names:
            ckpt_dir = os.path.join(matrices_dir, name)
            if not os.path.isdir(ckpt_dir):
                logger.error(f"Missing directory: {ckpt_dir}")
                return False

            pt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
            if len(pt_files) != 10:
                logger.error(f"{name}: expected 10 .pt files, found {len(pt_files)}")
                return False

            archive = os.path.join(ckpt_dir, "matrices.tar.gz")
            if not os.path.exists(archive):
                logger.error(f"{name}: missing matrices.tar.gz")
                return False

        # Spot-check: load first matrix from each checkpoint, verify reconstruction
        x_check = data[0][0]
        for name in checkpoint_names:
            state_dict = torch.load(
                os.path.join(weights_dir, f"{name}.pt"), weights_only=True
            )
            model.load_state_dict(state_dict)
            model.eval()
            forward_pass = model(x_check)

            mat = torch.load(
                os.path.join(matrices_dir, name, "sample_0.pt"),
                weights_only=True,
            )
            diff = torch.norm(forward_pass - mat.sum(1)).item()
            if diff >= 0.1:
                logger.error(f"{name}/sample_0.pt: reconstruction error {diff:.6e} >= 0.1")
                return False

        logger.info(f"All 3 checkpoints verified (10 samples + archive each)")
        logger.info("Phase 4: PASS")
        return True

    finally:
        shutil.rmtree(experiment_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results = {}

    # Phase 1
    results["smoke_test"] = phase1_smoke_test()

    # Phase 2
    optimal_sizes = phase2_calibration()
    results["calibration"] = True  # informational, always passes

    # Phase 3 — use optimal batch_size for (3, 32, 32)
    bs_32 = optimal_sizes.get((3, 32, 32), 1)
    results["dataset_computer"] = phase3_dataset_computer(bs_32)

    # Phase 4 — same batch_size
    results["experiment_runner"] = phase4_experiment_runner(bs_32)

    # Final summary
    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    for phase, passed in results.items():
        logger.info(f"  {phase:25s}: {'PASS' if passed else 'FAIL'}")
    for shape, bs in optimal_sizes.items():
        logger.info(f"  optimal batch_size {shape}: {bs}")
    logger.info("=" * 50)

    if all(results.values()):
        logger.info("All phases PASSED")
        sys.exit(0)
    else:
        logger.error("Some phases FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
