An implementation of the Faddeeva function and Voigt profile (real part of Faddeeva function) in `jax`. 

For imaginary Faddeva function argument `y` greater than 1e-10, accuracy $\vert \delta w/w_{ref}\vert$ relative to `scipy.special.wofz` should be 1e-4 or better. 

Uses approximations from Humlíček 1979 and 1982 and Weideman 1994, with help from Schreier 2017 and Zaghloul 2022. 
