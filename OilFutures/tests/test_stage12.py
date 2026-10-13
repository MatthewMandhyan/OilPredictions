# PROVIDED · Stage 12 checks: run with  pytest tests/test_stage12.py -v
# The first tests need only numpy; the rest are skipped automatically if PyTorch isn't installed.
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from models.neural_net import mixture_prob_above, softmax


def test_softmax_weights_sum_to_one():
    w = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 0.0]]))
    assert np.allclose(w.sum(axis=1), 1.0)
    assert (w >= 0).all() and np.isfinite(w).all()
    assert w[1, 0] == pytest.approx(0.5)


def test_single_component_matches_normal():
    x = np.array([-0.05, 0.0, 0.08])
    p = mixture_prob_above(x, np.array([1.0]), np.array([0.01]), np.array([0.04]))
    assert p == pytest.approx(norm.sf((x - 0.01) / 0.04))


def test_mixture_probabilities_monotonic_across_strikes():
    w, m, s = softmax(np.array([0.2, -1.0, 0.5])), np.array([-0.03, 0.0, 0.02]), np.array([0.02, 0.08, 0.04])
    p = mixture_prob_above(np.linspace(-0.3, 0.3, 61), w, m, s)
    assert (np.diff(p) < 0).all() and p[0] < 1 and p[-1] > 0


# ---- torch part -------------------------------------------------------------------

import models.neural_net as nnmod                                                # noqa: E402
from models.base import Contract                                                # noqa: E402
from tests.helpers import data_from_returns                                     # noqa: E402

torch = nnmod.torch
needs_torch = pytest.mark.skipif(torch is None, reason="PyTorch not installed (Stage 12 is optional)")


@needs_torch
def test_network_outputs_valid_mixture():
    torch.manual_seed(0)
    net = nnmod.MixtureDensityNet(n_inputs=4, n_components=3)
    w, m, s = net(torch.randn(16, 4))
    assert w.shape == m.shape == s.shape == (16, 3)
    assert torch.allclose(w.sum(dim=1), torch.ones(16))
    assert (s > 0).all()


@needs_torch
def test_nll_matches_scipy_for_one_component():
    w, m, s = torch.ones(3, 1), torch.zeros(3, 1), torch.full((3, 1), 0.5)
    y = torch.tensor([0.0, 0.5, -1.0])
    expected = -norm.logpdf([0.0, 0.5, -1.0], 0, 0.5).mean()
    assert nnmod.mixture_nll(w, m, s, y).item() == pytest.approx(expected, rel=1e-5)
    assert torch.isfinite(nnmod.mixture_nll(w, m, s, torch.tensor([50.0])))


@needs_torch
def test_neural_net_model_end_to_end():
    g = np.random.default_rng(0)
    r = g.normal(0, 0.3 / np.sqrt(252), 600)
    data, days = data_from_returns(r)
    as_of = pd.Timestamp(days[590]) + pd.Timedelta(hours=15, minutes=5)
    m = nnmod.NeuralNetModel(epochs=60, seed=0)
    m.fit(data, as_of)
    F = data.curve.settle("2099-12", as_of)
    probs = [m.predict(Contract(ticker=f"K{k}", contract_type="above", strike=F * k, settle_date=days[599],
                                futures_month="2099-12"), as_of).probability for k in (0.9, 0.95, 1.0, 1.05, 1.1)]
    assert all(0 <= p <= 1 for p in probs)
    assert all(a >= b for a, b in zip(probs, probs[1:])), "P(above K) must fall as K rises"
    assert 0.2 < probs[2] < 0.8
