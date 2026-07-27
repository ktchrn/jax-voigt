An implementation of the Faddeeva function and Voigt profile (real part of Faddeeva function) in `jax`. 

For imaginary Faddeva function argument `y` greater than 1e-10, accuracy $\vert \delta w/w_{ref}\vert$ relative to `scipy.special.wofz` should be 1e-4 or better. 

Uses approximations from Humlíček 1979 and 1982 and Weideman 1994, with help from Schreier 2017 and Zaghloul 2022. 

## Evaluation methods

By default (`method="auto"`) the Faddeeva function is evaluated with a
per-element region switch between continued-fraction, Weideman, and
Humlíček region 4 approximations, valid for all `y >= 0`.

For the small-damping absorption-line regime (`y <= 0.1`, i.e. damped
Lyman-type lines excluded), `method="humlicek4"` evaluates Humlíček
region 4 everywhere: fully vectorized, ~25x faster on CPU, and accurate
to ~1e-5 relative in the real part. Pass `method=` to `voigt_profile` /
`astro_voigt_profile` per call, or set a module-wide default:

```python
import jax_voigt
jax_voigt.set_default_method("humlicek4")
```

Set the default before building jitted computations; jax retraces on the
next call after a change.
