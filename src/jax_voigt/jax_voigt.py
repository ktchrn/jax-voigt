import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import cond, switch
from jax.lax import map as jmap
from jax import custom_jvp


__all__ = ["voigt_profile"]
_ISQ_PI = 1/np.sqrt(np.pi)


def voigt_profile(x, y):
    """
    Real part of the Faddeeva function. 
    Args:
        x (real array or scalar): real part of wofz argument
        y: (non-negative real array or scalar): imaginary part of wofz argument, assumed non-negative
    Returns:
        Voigt profile (aka real part of Faddeeva function) evaluated at x+1j*y
    """
    return _wofz(x, y)[0]


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
