import numpy as np
from .jax_voigt import astro_voigt_profile
import jax
import jax.numpy as jnp

import astropy.units as u
import astropy.constants as aco

c_in_cm_s = aco.c.to('cm/s').value
# assuming N(v) in units of s/cm**3, rest wavelength in cm
opacity_conversion_constant = np.pi*(aco.c*aco.a0*aco.alpha**2).to('cm**2/s').value

from collections import namedtuple


VoigtParams = namedtuple('VoigtParams', 'log10_N z b_cm_s')
LineParams = namedtuple('LineParams', 'wave_rest_cm f Gamma__s')
LineCompIndices = namedtuple('LineCompIndices', 'log10_N_idx z_idx b_idx')
SegmentParams = namedtuple('SegmentParams', 'line_def line_comp_inds wave_eval_cm LSF')
LinewiseSegmentParams = namedtuple('LinewiseSegmentParams', 'voigt_params_lines line_params wave_eval_cm')


def voigt_profile(voigt_params, line_params, wave_eval_cm):
    """
    Arguments:
        voigt_params (VoigtParams): log10_column density, centroid redshift, 
        and Gaussian width (sqrt(2)*standard deviation)
        line_params (LineParams): Rest wavelength, oscillator strength, and Lorentzian width
        wave_eval_cm (jax.numpy.array-like): Wavelength to evaluate Voigt profile at in cm
    Returns:
        Voigt profile

    """
    z0 = wave_eval_cm/line_params.wave_rest_cm - 1
    Gamma_nu0 = line_params.Gamma__s* (line_params.wave_rest_cm)/c_in_cm_s
    b_c = voigt_params.b_cm_s/c_in_cm_s

    norm = (line_params.wave_rest_cm * line_params.f * opacity_conversion_constant
            * 10**voigt_params.log10_N)
    return norm*astro_voigt_profile(voigt_params.z, b_c, Gamma_nu0, z0)


def components_to_lines(compwise_voigt_params, segment_params_sets):
    def single_expand(segment_params):
        lci = segment_params.line_comp_inds
        voigt_params_lines = VoigtParams(compwise_voigt_params.log10_N[lci.log10_N_idx],
                                         compwise_voigt_params.z[lci.z_idx],
                                         compwise_voigt_params.b_cm_s[lci.b_idx])
        return LinewiseSegmentParams(voigt_params_lines, segment_params.line_def, segment_params.wave_eval_cm)
    return [single_expand(segment_params) for segment_params in segment_params_sets]



vlt = jax.vmap(voigt_profile, [VoigtParams(None, None, None), LineParams(None, None, None), 0])



def _component_to_tau(linewise_segment_params):
	return vlt(*linewise_segment_params)

def components_to_tau(compwise_voigt_params, segment_params_sets):
    linewise_segment_params_sets = components_to_lines(compwise_voigt_params, segment_params_sets)
    segment_taus = [_component_to_tau(lsp) for lsp in linewise_segment_params_sets]
    return segment_taus



def _component_to_normed_flux(linewise_segment_params):
	segment_tau = jnp.sum(_component_to_tau(linewise_segment_params), axis=-1)
	return jnp.exp(-segment_tau)

def components_to_normed_fluxes(compwise_voigt_params, segment_params_sets):
    linewise_segment_params_sets = components_to_lines(compwise_voigt_params, segment_params_sets)
    segment_fluxes = [_component_to_normed_flux(lsp) for lsp in linewise_segment_params_sets]
    return segment_fluxes



def normed_fluxes_to_conv_fluxes(normed_fluxes, segment_params_sets):
    conv_fluxes = [1+jnp.convolve(nf-1, sp.LSF, mode='same') for nf, sp in zip(normed_fluxes, segment_params_sets)]
    return conv_fluxes



def _component_to_conv_flux(linewise_segment_params, segment_params):
	flux = _component_to_normed_flux(linewise_segment_params)
	return 1 + jnp.convolve(flux-1, segment_params.LSF, mode='same')

def components_to_conv_fluxes(compwise_voigt_params, segment_params_sets):
    linewise_segment_params_sets = components_to_lines(compwise_voigt_params, segment_params_sets)
    segment_fluxes = [_component_to_conv_flux(lsp, sp) for lsp, sp in zip(linewise_segment_params_sets, segment_params_sets)]
    return segment_fluxes
