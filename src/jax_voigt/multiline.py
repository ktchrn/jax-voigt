import numpy as np
from .jax_voigt import astro_voigt_profile
import jax
import jax.numpy as jnp

from collections import namedtuple

import astropy.units as u
import astropy.constants as aco

c_in_cm_s = aco.c.to('cm/s').value
c_in_km_s = aco.c.to('km/s').value

km_to_cm = u.km.to('cm')
Angstrom_to_cm = u.Angstrom.to('cm')
Angstrom_to_km = u.Angstrom.to('km')

# assuming N(v) in units of s/cm**3, rest wavelength in cm
opacity_conversion_constant = np.pi*(aco.c*aco.a0*aco.alpha**2*u.cm).to('cm**3/s').value


VoigtParams = namedtuple('VoigtParams', 'log10N z b_km_s')
LineParams = namedtuple('LineParams', 'wrest_A f Gamma__s')
LineCompIndices = namedtuple('LineCompIndices', 'log10N_idx z_idx b_idx')
SegmentParams = namedtuple('SegmentParams', 'line_def line_comp_inds wave LSF')
LinewiseSegmentParams = namedtuple('LinewiseSegmentParams', 'voigt_params_lines line_params wave')


def voigt_tau(voigt_params, line_params, wave_ax_A):
    z0 = wave_ax_A/line_params.wrest_A - 1
    G_nu0 = line_params.Gamma__s* (line_params.wrest_A*Angstrom_to_cm)/c_in_cm_s
    b_c = voigt_params.b_km_s/c_in_km_s
    
    norm = ((line_params.wrest_A*Angstrom_to_km) 
            * line_params.f*opacity_conversion_constant
            * 10**(voigt_params.log10N))
    return norm*astro_voigt_profile(voigt_params.z, b_c, G_nu0, z0)


def components_to_lines(compwise_voigt_params, segment_params_sets):
    def single_expand(segment_params):
        line_comp_inds = segment_params.line_comp_inds
        voigt_params_lines = VoigtParams(compwise_voigt_params.log10N[line_comp_inds.log10N_idx],
                                         compwise_voigt_params.z[line_comp_inds.z_idx],
                                         compwise_voigt_params.b_km_s[line_comp_inds.b_idx])
        return LinewiseSegmentParams(voigt_params_lines, segment_params.line_def, segment_params.wave)
    return [single_expand(segment_params) for segment_params in segment_params_sets]



vlt = jax.vmap(voigt_tau, [VoigtParams(None, None, None), LineParams(None, None, None), 0])



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
