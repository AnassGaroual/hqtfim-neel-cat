import sys
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if "code" in sys.modules:
    del sys.modules["code"]

from code import (  # noqa: E402
    v2,
    get_h_star,
    diagonalize_hierarchical,
    diagonalize_standard_1D,
)


def test_v2_basic():
    assert v2(0) == 100
    assert v2(1) == 0
    assert v2(2) == 1
    assert v2(4) == 2
    assert v2(12) == 2  # 12 = 3 * 2^2


def test_h_star_limits():
    # For large alpha, h* ~ 2^alpha / 2^(2 alpha) = 2^-alpha → 0
    h1 = get_h_star(1.5)
    h2 = get_h_star(3.0)
    assert h2 < h1
    assert 0.0 < h2 < 1.0


def test_small_hierarchical_N2():
    N = 2
    obs = diagonalize_hierarchical(N=N, J0=1.0, alpha=1.5, h=0.3)
    # Ground-state energy must be finite and negative
    assert obs["E0"] < 0.0
    # State is normalized: sum of Schmidt weights ≈ 1
    s0 = obs["schmidt_0"]
    s1 = obs["schmidt_1"]
    assert 0.9999 < s0**2 + s1**2 <= 1.0001


def test_standard_vs_hierarchical_same_N2():
    N = 2
    h = 0.4
    obs_h = diagonalize_hierarchical(N=N, J0=1.0, alpha=1.5, h=h)
    obs_s = diagonalize_standard_1D(N=N, J=1.0, h=h, pbc=True)

    # Energies must be finite and not NaN
    assert not math.isnan(obs_h["E0"])
    assert not math.isnan(obs_s["E0"])

    # Entropy must be between 0 and N/2 bits
    assert 0.0 <= obs_h["entropy"] <= N / 2
    assert 0.0 <= obs_s["entropy"] <= N / 2
