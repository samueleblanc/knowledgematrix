#!/usr/bin/env python
import unittest
import torch
import random
import psutil
import gc
from time import time

from knowledgematrix.models.sae import SAE
from knowledgematrix.autoencoder_computer import AutoencoderMatrixComputer

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)


def get_memory_usage() -> float:
    """
        Returns:
            Current memory usage in MB
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


class TestSAEReLU(unittest.TestCase):
    def test_knowledge_matrices(self) -> None:
        """Test SAE with ReLU activation."""
        for test_num in range(5):
            print(f"\n--------------------------------")
            print(f"ReLU SAE - Test {test_num + 1}/5:")

            gc.collect()
            start_mem = get_memory_usage()
            start_time = time()

            d_model = random.randint(16, 64)
            d_hidden = random.randint(d_model, d_model * 4)

            model = SAE(d_model=d_model, d_hidden=d_hidden, activation="relu").to(DEVICE)
            model.eval()

            x = torch.rand(d_model, 1, 1)
            forward_pass = model(x)

            batch_size = random.randint(1, 16)
            computer = AutoencoderMatrixComputer(model, batch_size=batch_size)
            encoder_mat, decoder_mat, total_mat = computer.forward(x)

            # Verify total knowledge matrix
            diff_total = torch.norm(forward_pass - total_mat.sum(1)).item()

            # Verify encoder knowledge matrix
            encoder = model.get_encoder()
            encoder.eval()
            encoder_out = encoder(x)
            diff_encoder = torch.norm(encoder_out - encoder_mat.sum(1)).item()

            # Verify decoder knowledge matrix
            latent = encoder_out.reshape(model.latent_shape)
            decoder = model.get_decoder()
            decoder.eval()
            decoder_out = decoder(latent)
            diff_decoder = torch.norm(decoder_out - decoder_mat.sum(1)).item()

            end_time = time()
            end_mem = get_memory_usage()

            self.assertAlmostEqual(diff_total, 0, delta=0.1,
                msg=f"Total: rep and forward_pass differ by {diff_total}.")
            self.assertAlmostEqual(diff_encoder, 0, delta=0.1,
                msg=f"Encoder: rep and forward_pass differ by {diff_encoder}.")
            self.assertAlmostEqual(diff_decoder, 0, delta=0.1,
                msg=f"Decoder: rep and forward_pass differ by {diff_decoder}.")

            print(f"  d_model={d_model}, d_hidden={d_hidden}, batch_size={batch_size}")
            print(f"  Diff total={diff_total:.2e}, encoder={diff_encoder:.2e}, decoder={diff_decoder:.2e}")
            print(f"  Time: {end_time-start_time:.4f}s  Memory: {end_mem-start_mem:.4f}MB")
            print(f"--------------------------------")


class TestSAEJumpReLU(unittest.TestCase):
    def test_knowledge_matrices(self) -> None:
        """Test SAE with JumpReLU activation."""
        for test_num in range(5):
            print(f"\n--------------------------------")
            print(f"JumpReLU SAE - Test {test_num + 1}/5:")

            gc.collect()
            start_mem = get_memory_usage()
            start_time = time()

            d_model = random.randint(16, 64)
            d_hidden = random.randint(d_model, d_model * 4)
            thresholds = torch.rand(d_hidden) * 0.5

            model = SAE(
                d_model=d_model, d_hidden=d_hidden,
                activation="jumprelu", thresholds=thresholds
            ).to(DEVICE)
            model.eval()

            x = torch.rand(d_model, 1, 1)
            forward_pass = model(x)

            batch_size = random.randint(1, 16)
            computer = AutoencoderMatrixComputer(model, batch_size=batch_size)
            encoder_mat, decoder_mat, total_mat = computer.forward(x)

            diff_total = torch.norm(forward_pass - total_mat.sum(1)).item()

            encoder = model.get_encoder()
            encoder.eval()
            encoder_out = encoder(x)
            diff_encoder = torch.norm(encoder_out - encoder_mat.sum(1)).item()

            latent = encoder_out.reshape(model.latent_shape)
            decoder = model.get_decoder()
            decoder.eval()
            decoder_out = decoder(latent)
            diff_decoder = torch.norm(decoder_out - decoder_mat.sum(1)).item()

            end_time = time()
            end_mem = get_memory_usage()

            self.assertAlmostEqual(diff_total, 0, delta=0.1,
                msg=f"Total: rep and forward_pass differ by {diff_total}.")
            self.assertAlmostEqual(diff_encoder, 0, delta=0.1,
                msg=f"Encoder: rep and forward_pass differ by {diff_encoder}.")
            self.assertAlmostEqual(diff_decoder, 0, delta=0.1,
                msg=f"Decoder: rep and forward_pass differ by {diff_decoder}.")

            print(f"  d_model={d_model}, d_hidden={d_hidden}, batch_size={batch_size}")
            print(f"  Diff total={diff_total:.2e}, encoder={diff_encoder:.2e}, decoder={diff_decoder:.2e}")
            print(f"  Time: {end_time-start_time:.4f}s  Memory: {end_mem-start_mem:.4f}MB")
            print(f"--------------------------------")


class TestSAETopK(unittest.TestCase):
    def test_knowledge_matrices(self) -> None:
        """Test SAE with TopK activation."""
        for test_num in range(5):
            print(f"\n--------------------------------")
            print(f"TopK SAE - Test {test_num + 1}/5:")

            gc.collect()
            start_mem = get_memory_usage()
            start_time = time()

            d_model = random.randint(16, 64)
            d_hidden = random.randint(d_model, d_model * 4)
            k = random.randint(1, max(1, d_hidden // 4))

            model = SAE(
                d_model=d_model, d_hidden=d_hidden,
                activation="topk", k=k
            ).to(DEVICE)
            model.eval()

            x = torch.rand(d_model, 1, 1)
            forward_pass = model(x)

            batch_size = random.randint(1, 16)
            computer = AutoencoderMatrixComputer(model, batch_size=batch_size)
            encoder_mat, decoder_mat, total_mat = computer.forward(x)

            diff_total = torch.norm(forward_pass - total_mat.sum(1)).item()

            encoder = model.get_encoder()
            encoder.eval()
            encoder_out = encoder(x)
            diff_encoder = torch.norm(encoder_out - encoder_mat.sum(1)).item()

            latent = encoder_out.reshape(model.latent_shape)
            decoder = model.get_decoder()
            decoder.eval()
            decoder_out = decoder(latent)
            diff_decoder = torch.norm(decoder_out - decoder_mat.sum(1)).item()

            end_time = time()
            end_mem = get_memory_usage()

            self.assertAlmostEqual(diff_total, 0, delta=0.1,
                msg=f"Total: rep and forward_pass differ by {diff_total}.")
            self.assertAlmostEqual(diff_encoder, 0, delta=0.1,
                msg=f"Encoder: rep and forward_pass differ by {diff_encoder}.")
            self.assertAlmostEqual(diff_decoder, 0, delta=0.1,
                msg=f"Decoder: rep and forward_pass differ by {diff_decoder}.")

            print(f"  d_model={d_model}, d_hidden={d_hidden}, k={k}, batch_size={batch_size}")
            print(f"  Diff total={diff_total:.2e}, encoder={diff_encoder:.2e}, decoder={diff_decoder:.2e}")
            print(f"  Time: {end_time-start_time:.4f}s  Memory: {end_mem-start_mem:.4f}MB")
            print(f"--------------------------------")


class TestSAELoadWeights(unittest.TestCase):
    def test_load_weights_with_bdec_folding(self) -> None:
        """Test that load_weights correctly folds b_dec into encoder bias."""
        d_model = 32
        d_hidden = 64

        W_enc = torch.randn(d_hidden, d_model)
        b_enc = torch.randn(d_hidden)
        W_dec = torch.randn(d_model, d_hidden)
        b_dec = torch.randn(d_model)

        model = SAE(d_model=d_model, d_hidden=d_hidden, activation="relu")
        model.load_weights(W_enc, b_enc, W_dec, b_dec)
        model.eval()

        # Test that the SAE computes f(x) = sigma(W_enc @ (x - b_dec) + b_enc)
        # and x_hat = W_dec @ f(x) + b_dec
        x_flat = torch.randn(d_model)
        expected_hidden = torch.relu(W_enc @ (x_flat - b_dec) + b_enc)
        expected_output = W_dec @ expected_hidden + b_dec

        x = x_flat.reshape(d_model, 1, 1)
        actual_output = model(x).flatten()

        diff = torch.norm(expected_output - actual_output).item()
        self.assertAlmostEqual(diff, 0, delta=1e-6,
            msg=f"load_weights b_dec folding error: {diff}")
        print(f"\nload_weights b_dec folding diff: {diff:.2e}")

        # Verify all 3 knowledge matrices after loading weights
        batch_size = 8
        computer = AutoencoderMatrixComputer(model, batch_size=batch_size)
        encoder_mat, decoder_mat, total_mat = computer.forward(x)

        diff_total = torch.norm(actual_output - total_mat.sum(1)).item()

        encoder = model.get_encoder()
        encoder.eval()
        encoder_out = encoder(x)
        diff_encoder = torch.norm(encoder_out - encoder_mat.sum(1)).item()

        latent = encoder_out.reshape(model.latent_shape)
        decoder = model.get_decoder()
        decoder.eval()
        decoder_out = decoder(latent)
        diff_decoder = torch.norm(decoder_out - decoder_mat.sum(1)).item()

        self.assertAlmostEqual(diff_total, 0, delta=0.1,
            msg=f"Total: rep and forward_pass differ by {diff_total}.")
        self.assertAlmostEqual(diff_encoder, 0, delta=0.1,
            msg=f"Encoder: rep and forward_pass differ by {diff_encoder}.")
        self.assertAlmostEqual(diff_decoder, 0, delta=0.1,
            msg=f"Decoder: rep and forward_pass differ by {diff_decoder}.")

        print(f"  Knowledge matrices — total={diff_total:.2e}, encoder={diff_encoder:.2e}, decoder={diff_decoder:.2e}")


if __name__ == "__main__":
    unittest.main()
