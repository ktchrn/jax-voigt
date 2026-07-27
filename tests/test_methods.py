"""Accuracy and dispatch tests for the Faddeeva method option.

scipy.special.wofz is the reference; scipy is a test-only dependency.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import wofz

import jax_voigt
from jax_voigt import astro_voigt_profile, set_default_method, voigt_profile


def wofz_real(x, y):
    return np.real(wofz(np.asarray(x) + 1j * np.asarray(y)))


def rel_err(got, ref):
    return np.max(np.abs(np.asarray(got) - ref) / np.abs(ref))


X = np.linspace(-300.0, 300.0, 4001)


@pytest.mark.parametrize("y", [1e-4, 1e-3, 1e-2, 0.05, 0.1])
def test_humlicek4_accuracy_in_domain(y):
    ref = wofz_real(X, np.full_like(X, y))
    got = voigt_profile(jnp.asarray(X), jnp.full(X.size, y), method="humlicek4")
    assert rel_err(got, ref) < 5e-5


@pytest.mark.parametrize("y", [1e-4, 1e-2, 0.5, 5.0, 50.0])
def test_auto_accuracy_everywhere(y):
    ref = wofz_real(X, np.full_like(X, y))
    got = voigt_profile(jnp.asarray(X), jnp.full(X.size, y), method="auto")
    assert rel_err(got, ref) < 2e-4


def test_default_method_is_auto():
    y = 5.0  # outside the humlicek4 domain: only "auto" gets this right
    ref = wofz_real(X, np.full_like(X, y))
    got = voigt_profile(jnp.asarray(X), jnp.full(X.size, y))
    assert rel_err(got, ref) < 2e-4


def test_set_default_method():
    try:
        set_default_method("humlicek4")
        ref = wofz_real(X, np.full_like(X, 1e-3))
        got = voigt_profile(jnp.asarray(X), jnp.full(X.size, 1e-3))
        assert rel_err(got, ref) < 5e-5
    finally:
        set_default_method("auto")


def test_set_default_method_rejects_unknown():
    with pytest.raises(ValueError, match="method must be one of"):
        set_default_method("exact")
    with pytest.raises(ValueError, match="method must be one of"):
        voigt_profile(jnp.asarray([0.0]), jnp.asarray([1e-3]), method="exact")


def test_astro_voigt_profile_threads_method():
    # a typical metal-line configuration: the two methods must agree
    z0 = jnp.linspace(-1e-3, 1e-3, 2001)
    kwargs = dict(centroid_redshift=1e-4, b_c=3e-5, Γ_ν0=1e-8, eval_redshift=z0)
    vp_auto = astro_voigt_profile(**kwargs, method="auto")
    vp_h4 = astro_voigt_profile(**kwargs, method="humlicek4")
    assert rel_err(vp_h4, np.asarray(vp_auto)) < 5e-5


def test_humlicek4_gradients_match_auto():
    # pointwise dK/dx and dK/dy at places where they are O(1); the custom
    # jvp identities are shared, so this checks the (K, L) each method feeds
    # them and that gradients flow finitely through the humlicek4 path
    points = jnp.asarray([0.5, 1.0, 2.0, 5.0])
    y = 1e-2

    for method in ("humlicek4",):
        dx_m = jax.vmap(jax.grad(
            lambda x: voigt_profile(x, y, method=method)))(points)
        dy_m = jax.vmap(jax.grad(
            lambda x, yy: voigt_profile(x, yy, method=method), argnums=1),
            in_axes=(0, None))(points, y)
        dx_a = jax.vmap(jax.grad(
            lambda x: voigt_profile(x, y, method="auto")))(points)
        dy_a = jax.vmap(jax.grad(
            lambda x, yy: voigt_profile(x, yy, method="auto"), argnums=1),
            in_axes=(0, None))(points, y)
        assert np.all(np.isfinite(dx_m)) and np.all(np.isfinite(dy_m))
        np.testing.assert_allclose(np.asarray(dx_m), np.asarray(dx_a), rtol=1e-3)
        np.testing.assert_allclose(np.asarray(dy_m), np.asarray(dy_a), rtol=1e-3)


def test_humlicek4_at_y_zero_is_gaussian():
    x = jnp.linspace(-5.0, 5.0, 101)
    got = voigt_profile(x, jnp.zeros(101), method="humlicek4")
    np.testing.assert_allclose(np.asarray(got), np.exp(-np.asarray(x) ** 2),
                               atol=2e-5)
