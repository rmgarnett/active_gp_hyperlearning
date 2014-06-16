Active GP Hyperparameter Learning
=================================

This is a MATLAB implementation of the method for actively learning GP
hyperparameters described in
> Garnett, R., Osborne, M., and Hennig, P. Active Learning of Linear
> Embeddings for Gaussian Processes. (2014). 30th Conference on
> Uncertainty in Artificial Intelligence (UAI 2014).

Given a GP model on a function *f*:

![p(f | \theta) = GP(f; mu(x; \theta), K(x, x'; \theta))][1]

this routine sequentially chooses a sequence of locations *X* =
{*x*<sub>*i*</sub>} to make observations with the goal of learning the
GP hyperparameters *&theta;* as quickly as possible. This is done by
maintaining a probabilistic belief *p*(*&theta;* | *D*) and selecting
each observation location by maximizing the Bayesian active learning
by disagreement (BALD) criterion described in

> N. Houlsby, F. Huszar, Z. Ghahramani, and M. Lengyel. Bayesian
> Active Learning for Classification and Preference
> Learning. (2011). arXiv preprint arXiv:1112.5745 [stat.ML].

This implementation uses the approximation to BALD described in the
Garnett, et al. paper above, which relies on the "marginal GP" (MGP)
method for approximate GP hyperparameter marginalization.

The main entrypoint is `learn_gp_hyperparameters.m`. See `demo/demo.m`
for a simple example usage.

Dependencies
------------

This code is written to be interoperable with the GPML MATLAB
toolbox, available here:

  http://www.gaussianprocess.org/gpml/code/matlab/doc/

The GPML toolbox must be in your MATLAB path for this function to
work. This function also depends on the gpml_extensions repository,
available here:

  https://github.com/rmgarnett/gpml_extensions/

as well as the marginal GP (MGP) implementation available here:

  https://github.com/rmgarnett/mgp/

Both must be in your MATLAB path. Finally, the optimization of
the GP log posterior requires Mark Schmidt's minFunc function:

  http://www.di.ens.fr/~mschmidt/Software/minFunc.html

[1]: http://latex.codecogs.com/svg.latex?p(f%20%5Cmid%20%5Ctheta)%20%3D%20%5Cmathcal%7BGP%7D%5Cbigl(f%3B%20%5Cmu(x%3B%20%5Ctheta)%2C%20K(x%2C%20x%27%3B%20%5Ctheta)%5Cbigr)
