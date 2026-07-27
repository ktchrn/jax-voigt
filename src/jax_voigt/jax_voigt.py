import numpy as np
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import cond, switch
from jax.lax import map as jmap
from jax import custom_jvp
import astropy.constants as aco

c_in_cm_s = aco.c.to('cm/s').value


__all__ = ["voigt_profile", "astro_voigt_profile", "set_default_method"]
_ISQ_PI = 1/np.sqrt(np.pi)

_METHODS = ("auto", "humlicek4")
_DEFAULT_METHOD = "auto"


def set_default_method(method):
    """
    Set the Faddeeva evaluation method used when none is passed explicitly.
    Arguments:
        method (str): one of
            "auto" -- region-switched evaluation (continued fractions,
                Weideman, Humlicek region 4), valid for all y >= 0. The
                per-element dispatch makes it markedly slower than a single
                branch.
            "humlicek4" -- Humlicek (1982) region 4 only, fully vectorized.
                ~25x faster than "auto" and accurate to ~1e-5 relative for
                y <= 0.1 (the unsaturated-to-moderately-damped absorption
                line regime); NOT valid at larger y.
    Functions are retraced by jax on their next call after a change; set the
    default before building jitted computations.
    """
    global _DEFAULT_METHOD
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    _DEFAULT_METHOD = method


def _resolve_wofz(method):
    if method is None:
        method = _DEFAULT_METHOD
    if method == "auto":
        return _wofz
    if method == "humlicek4":
        return _wofz_hum4
    raise ValueError(f"method must be one of {_METHODS}, got {method!r}")


def voigt_profile(x, y, method=None):
    """
    Real part of the Faddeeva function.
    Arguments:
        x (real array or scalar): real part of wofz argument
        y: (non-negative real array or scalar): imaginary part of wofz argument, assumed non-negative
    Keyword arguments:
        method (str or None): Faddeeva evaluation method; None uses the
            module default (see set_default_method).
    Returns:
        Voigt profile (aka real part of Faddeeva function) evaluated at x+1j*y
    """
    return _resolve_wofz(method)(x, y)[0]


def astro_voigt_profile(centroid_redshift, b_c, Γ_ν0, eval_redshift, speed_of_light=c_in_cm_s,
                        method=None):
    """
    Evaluate the Voigt profile function starting from astronomy-convention quantities.
    Arguments:
        centroid_redshift (float-like): Redshift of the profile center relative
                                        to the line rest frequency
        Γ_ν0 (float-like): Damping constant Γ=γ*4π in units of the line rest frequency
        b_c (float-like): Doppler broadening parameter b=sqrt(2)*σ in units of the speed of light
        eval_redshift (float-like):
    Keyword arguments:
        speed_of_light (float-like): c in units of your choice, default is in cm/s
        method (str or None): Faddeeva evaluation method; None uses the
            module default (see set_default_method).
    Returns:
        Voigt profile in units of 1/[speed_of_light units]
    """
    c_b = 1/b_c
    x = c_b*(eval_redshift - centroid_redshift)/(1+eval_redshift)
    y = c_b*Γ_ν0/(4*np.pi)
    vp = _ISQ_PI*(c_b/speed_of_light) * voigt_profile(x, y, method=method)
    return vp


@custom_jvp
def _wofz(x, y):
    """
    y is assumed to be positive, 
    """
    z = x + 1j*y
    z_flat = jnp.ravel(z)
    w_flat = jmap(_wofz_single, z_flat)
    w = jnp.reshape(w_flat, jnp.shape(z))
    K = jnp.real(w)
    L = jnp.imag(w)
    return K, L

@_wofz.defjvp
def _wofz_jvp(primals, tangents):
    x, y = primals
    xdot, ydot = tangents
    K, L = _wofz(x, y)
    dKdx = - 2 * (x * K - y * L)
    dKdy = 2 * (x * L + y * K) - 2.*_ISQ_PI
    dLdx = -1*dKdy
    dLdy = dKdx
    return (K, L), (xdot*dKdx+ydot*dKdy, xdot*dLdx+ydot*dLdy)


@custom_jvp
def _wofz_hum4(x, y):
    """
    Humlicek region 4 everywhere: vectorized, no per-element dispatch.
    Valid for y <= ~0.1 at ~1e-5 relative accuracy in the real part.
    """
    w = _humlicek4(x + 1j*y)
    return jnp.real(w), jnp.imag(w)


@_wofz_hum4.defjvp
def _wofz_hum4_jvp(primals, tangents):
    x, y = primals
    xdot, ydot = tangents
    K, L = _wofz_hum4(x, y)
    dKdx = - 2 * (x * K - y * L)
    dKdy = 2 * (x * L + y * K) - 2.*_ISQ_PI
    dLdx = -1*dKdy
    dLdy = dKdx
    return (K, L), (xdot*dKdx+ydot*dKdy, xdot*dLdx+ydot*dLdy)


def _wofz_single(z):
    x = jnp.real(z)
    y = jnp.imag(z)
    s = jnp.abs(x) + y
    
    # following suggestion of Zaghloul 2022, use Humlicek reg. 4 for region near y=0 out to 
    # where single term continued fraction becomes OK.
    # just going for 1e-4 δw/wref here, with wref being scipy wofz, so OK to use
    # hum4 up to y = 0.1 rather than 1e-6 as stated in Zaghloul 2022
    hum4_y = 10**-1.0
    
    reg1 = ((9<s) & (s<15) & (y>hum4_y)).astype(int)
    reg2 = ((s<=9) & (y>hum4_y)).astype(int)
    reg3 = ((s<15) & (y<=hum4_y)).astype(int)
    index = reg1*1 + reg2*2 + reg3*3
    w = switch(index, [_cf1, _cf3, _weid, _humlicek4], z)

    # continued fraction does fine for the imaginary part but has trouble
    # with the real part, patch in Gaussian
    # yes I know it would be better to use the Dawson function instead
    w = cond(y>0, lambda _: w, lambda _: jnp.exp(-x**2)+1j*jnp.imag(w), w)
    return w



"""
continued fraction expansions
"""
def _cf1 (z):
    w = 1j*_ISQ_PI / (z - 0.5/z)
    return w


def _cf3(z):
    w = 1j*_ISQ_PI / (z - 0.5/(z-1.0/(z-1.5/z)))
    return w



def _calc_weideman_coeffs(N):
    """
    polynomial coefficients for Weideman 1994 rational approximation
    """
    M = 2*N
    M2 = 2*M
    k = np.arange(-M+1, M)

    L = np.sqrt(N/np.sqrt(2))
    theta = k*np.pi/M
    t = L*np.tan(theta/2)

    f = np.exp(-t**2) * (L**2+t**2)
    f = np.concatenate([[0], f])

    a = np.fft.fft(np.fft.fftshift(f)).real/M2
    a = np.flip(a[1:N+1])
    return a


_WEID_N = 16
_WEID_A = _calc_weideman_coeffs(_WEID_N)
_WEID_L = np.sqrt(_WEID_N/np.sqrt(2))


def _weid(z):
    """
    Weideman 2014 rational approximation; number of terms is set by _WEID_N variable in jax_voigt.py
    """
    iz = 1j*z
    lpiz = _WEID_L + iz
    lmiz = _WEID_L - iz
    Z = lpiz /lmiz
    
    p = jnp.polyval(_WEID_A, Z)
    return (_ISQ_PI  +  2.0 * p / lmiz)  /  lmiz


def _humlicek4 (z):
    """
    Humlicek 1982 region 4
    Code adapted from Schreier 2018, changing coefficients to agree with original values from Humlicek 1982
    """
    t = -1j*z 
    u = t*t
    nom = t*(36183.31-u*(3321.9905-u*(1540.787-u*(219.0313-u*(35.76683-u*(1.320522-u*.56419))))))
    den = 32066.6-u*(24322.84-u*(9022.228-u*(2186.181-u*(364.2191-u*(61.57037-u*(1.841439-u))))))
    w  = jnp.exp(u) - nom/den
    return w
