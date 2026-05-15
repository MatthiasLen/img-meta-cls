"""
Tests for :mod:`IMC.nn.sparse_metadata_encoder`.

Test organisation
-----------------
1. **Smoke / forward pass** — Basic shape and dtype checks.
2. **NaN handling** — All-NaN, partial-NaN, all-valid, all-NaN batch.
3. **learn_missing_emb** — Behaviour with the learnable missing embedding flag.
4. **Backward-compatibility** — Default encoder uses a zero buffer (not a
   trainable parameter) for the missing embedding.
5. **Parameter counts** — ``learn_missing_emb=True`` adds exactly
   ``index_embed_dim`` trainable parameters vs the default.

Run the full suite::

    pytest tests/test_sparse_metadata_encoder.py -v
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from IMC.nn.sparse_metadata_encoder import SparseMetadataEncoder

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_B, _S, _F = 4, 8, 10
_DEFAULT_KWARGS = dict(num_features=_F, out_dim=128, scalar_modulation=False)


def _make_encoder(**extra) -> SparseMetadataEncoder:
    enc = SparseMetadataEncoder(**_DEFAULT_KWARGS, **extra)
    enc.eval()
    return enc


def _rand_input(nan_prob: float = 0.8, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    x = torch.randn(_B, _S, _F)
    x[torch.rand_like(x) < nan_prob] = float("nan")
    x[:, :, _F // 2] = 0.0  # zero is a valid (non-NaN) value
    return x


# ---------------------------------------------------------------------------
# 1. Smoke / forward pass
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_output_shape(self):
        enc = _make_encoder()
        x = _rand_input()
        z = enc(x)
        assert z.shape == (_B, 128)

    def test_output_is_finite(self):
        enc = _make_encoder()
        x = _rand_input(nan_prob=0.5)
        z = enc(x)
        assert torch.isfinite(z).all(), "Output contains NaN or Inf"

    def test_no_nan_input(self):
        enc = _make_encoder()
        x = torch.randn(_B, _S, _F)
        z = enc(x)
        assert z.shape == (_B, 128)
        assert torch.isfinite(z).all()


# ---------------------------------------------------------------------------
# 2. NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    """Encoder must handle all combinations of NaN patterns gracefully."""

    def test_partial_nan(self):
        enc = _make_encoder()
        x = _rand_input(nan_prob=0.8)
        z = enc(x)
        assert z.shape == (_B, 128)
        assert torch.isfinite(z).all()

    def test_all_nan_rows(self):
        """Rows where every feature is NaN should produce a finite output."""
        enc = _make_encoder()
        x = torch.full((_B, _S, _F), float("nan"))
        z = enc(x)
        assert z.shape == (_B, 128)
        assert torch.isfinite(z).all()

    def test_all_nan_single_sample(self):
        """Single batch item that is entirely NaN."""
        enc = _make_encoder()
        x = _rand_input(nan_prob=0.5)
        x[0] = float("nan")  # force first sample fully NaN
        z = enc(x)
        assert torch.isfinite(z).all()

    def test_mixed_all_nan_and_normal(self):
        """Batch with some all-NaN samples and some normal samples."""
        enc = _make_encoder()
        x = _rand_input(nan_prob=0.3)
        x[1] = float("nan")
        x[3] = float("nan")
        z = enc(x)
        assert z.shape == (_B, 128)
        assert torch.isfinite(z).all()


# ---------------------------------------------------------------------------
# 3. learn_missing_emb flag
# ---------------------------------------------------------------------------

class TestLearnMissingEmb:
    def test_output_shape_with_learned_emb(self):
        enc = _make_encoder(learn_missing_emb=True)
        x = _rand_input()
        z = enc(x)
        assert z.shape == (_B, 128)

    def test_output_finite_with_learned_emb(self):
        enc = _make_encoder(learn_missing_emb=True)
        x = _rand_input(nan_prob=0.8)
        z = enc(x)
        assert torch.isfinite(z).all()

    def test_all_nan_with_learned_emb(self):
        enc = _make_encoder(learn_missing_emb=True)
        x = torch.full((_B, _S, _F), float("nan"))
        z = enc(x)
        assert torch.isfinite(z).all()

    def test_missing_emb_is_trainable_parameter(self):
        enc = _make_encoder(learn_missing_emb=True)
        param_names = {n for n, _ in enc.named_parameters()}
        assert "missing_emb" in param_names, (
            "missing_emb should be a trainable parameter when learn_missing_emb=True"
        )

    def test_missing_emb_receives_gradients(self):
        # Construct encoder in train mode so gradients are computed
        enc = SparseMetadataEncoder(**_DEFAULT_KWARGS, learn_missing_emb=True)
        enc.train()
        # Use an all-NaN batch so that the missing embedding is exercised
        x = torch.full((_B, _S, _F), float("nan"))
        z = enc(x)
        loss = z.sum()
        loss.backward()
        assert enc.missing_emb.grad is not None, "missing_emb should receive gradients"
        assert torch.isfinite(enc.missing_emb.grad).all(), "missing_emb.grad should be finite"

    def test_missing_emb_not_trainable_by_default(self):
        enc = _make_encoder(learn_missing_emb=False)
        param_names = {n for n, _ in enc.named_parameters()}
        assert "missing_emb" not in param_names, (
            "missing_emb should NOT be a trainable parameter when learn_missing_emb=False"
        )


# ---------------------------------------------------------------------------
# 4. Backward-compatibility: default missing_emb buffer is zeros
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_missing_emb_buffer_is_zeros(self):
        enc = _make_encoder(learn_missing_emb=False)
        buf = dict(enc.named_buffers()).get("missing_emb")
        assert buf is not None, "missing_emb buffer not found"
        assert torch.all(buf == 0), "missing_emb buffer should be zero for backward compat"


# ---------------------------------------------------------------------------
# 5. Parameter counts
# ---------------------------------------------------------------------------

class TestParameterCounts:
    def _n_trainable(self, enc: SparseMetadataEncoder) -> int:
        return sum(p.numel() for p in enc.parameters() if p.requires_grad)

    def test_learned_has_more_params(self):
        default = _make_encoder(learn_missing_emb=False)
        learned = _make_encoder(learn_missing_emb=True)
        delta = self._n_trainable(learned) - self._n_trainable(default)
        assert delta > 0, "learn_missing_emb=True should add trainable parameters"

    def test_delta_equals_index_embed_dim(self):
        """The extra params should equal the size of the missing embedding."""
        default = _make_encoder(learn_missing_emb=False)
        learned = _make_encoder(learn_missing_emb=True)
        expected_delta = learned.missing_emb.numel()
        delta = self._n_trainable(learned) - self._n_trainable(default)
        assert delta == expected_delta, (
            f"Expected delta={expected_delta}, got {delta}"
        )
