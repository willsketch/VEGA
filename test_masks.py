"""
Tests for mask randomization functions in Vega-Runs.py.
Run with: python test_masks.py
"""
import numpy as np
import importlib
# TODO: rename Vega-Runs.py to vega_runs.py to allow clean Python imports
vega_runs = importlib.import_module("Vega-Runs")
randomize_mask = vega_runs.randomize_mask
degree_preserving_mask = vega_runs.degree_preserving_mask


def make_test_mask(n_genes=100, n_modules=20, density=0.1, seed=0):
    """Create a random binary mask for testing."""
    rng = np.random.default_rng(seed)
    mask = (rng.random((n_genes, n_modules)) < density).astype(float)
    return mask


def test_true_mask(mask):
    assert mask.ndim == 2, "Mask should be 2D"
    assert set(np.unique(mask)).issubset({0.0, 1.0}), "Mask should be binary"
    print("true mask: binary and 2D")


def test_random_mask(true_mask, rand_mask, dp_mask):
    assert set(np.unique(rand_mask)).issubset({0.0, 1.0}), "Random mask should be binary"
    assert rand_mask.shape == true_mask.shape, "Shape mismatch"
    assert int(rand_mask.sum()) == int(true_mask.sum()), \
        f"Edge count mismatch: {int(rand_mask.sum())} vs {int(true_mask.sum())}"
    assert not np.array_equal(rand_mask, true_mask), "Random mask is identical to true mask"
    assert not np.array_equal(rand_mask, dp_mask), "Random mask is identical to degree-preserving mask"
    print("random mask: binary, same shape, same edge count, different from true and degree-preserving")


def test_degree_preserving_mask(true_mask, dp_mask):
    assert set(np.unique(dp_mask)).issubset({0.0, 1.0}), "DP mask should be binary"
    assert dp_mask.shape == true_mask.shape, "Shape mismatch"
    assert np.array_equal(dp_mask.sum(axis=1), true_mask.sum(axis=1)), \
        "Row sums differ - gene degrees not preserved"
    assert np.array_equal(dp_mask.sum(axis=0), true_mask.sum(axis=0)), \
        "Column sums differ - module sizes not preserved"
    assert not np.array_equal(dp_mask, true_mask), "DP mask is identical to true mask"
    print("degree_preserving mask: binary, same shape, row sums preserved, col sums preserved, different from true")


if __name__ == "__main__":
    print("Generating test mask (100 genes × 20 modules, density=0.1)...")
    true_mask = make_test_mask()
    print(f"Total edges: {int(true_mask.sum())}")

    print("\nTesting true mask...")
    test_true_mask(true_mask)

    print("\nTesting degree-preserving mask...")
    dp_mask = degree_preserving_mask(true_mask, seed=42)
    test_degree_preserving_mask(true_mask, dp_mask)

    print("\nTesting random mask...")
    rand_mask = randomize_mask(true_mask, seed=42)
    test_random_mask(true_mask, rand_mask, dp_mask)

    print("\nAll tests passed.")