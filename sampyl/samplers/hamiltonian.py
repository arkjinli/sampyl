"""
sampyl.samplers.hamiltonian_metric
~~~~~~~~~~~~~~~~~~~~

Module implementing Hamiltonian MCMC sampler with Euclidean metric (constant
mass matrix), with code reproduced from Matt Graham's HMC code
https://github.com/matt-graham/hmc

:copyright: (c) 2017 by Jinli Hu.
:license: Apache2, see LICENSE for more details.

"""


from __future__ import division

from ..core import np
from ..state import State
from .base import Sampler
from ..model import Model
import scipy.linalg as la


class Hamiltonian(Sampler):
    def __init__(self, logp, start, mass=None, step_size=1, n_steps=5, **kwargs):

        """ Hamiltonian MCMC sampler with Euclidean metric defined by a constant
            mass matrix. Uses the gradient of log P(theta) to make informed
            proposals.

            Arguments
            ----------
            logp: function
                log P(X) function for sampling distribution
            start: dict
                Dictionary of starting state for the sampler. Should have one
                element for each argument of logp. So, if logp = f(x, y), then
                start = {'x': x_start, 'y': y_start}

            Keyword Arguments
            -----------------
            mass: 2d array
                A constant mass matrix (2d array), which is usually fed by the
                Hessian at the MAP estimate of the position variables
            grad_logp: function or list of functions
                Functions that calculate grad log P(theta). Pass functions
                here if you don't want to use autograd for the gradients. If
                logp has multiple parameters, grad_logp must be a list of
                gradient functions w.r.t. each parameter in logp.
            step_size: float
                Step size for the deterministic proposals.
            n_steps: int
                Number of deterministic steps to take for each proposal.
            """

        super(Hamiltonian, self).__init__(logp, start, **kwargs)

        if mass is None:
            self.mass = default_mass(self.state, self.scale, self.conditional)
        else:
            assert np.all(mass.T - mass <= 1e-6), 'mass matrix is asymmetric'
            assert self.state.tovector().size == mass.shape[0], \
                'mass matrix dimensionality does not match states'
            self.mass = mass

        self.dim = self.mass.shape[0]
        self.mass_chol = la.cholesky(self.mass, lower=True)
        self.step_size = step_size / self.dim**(1/4)
        self.n_steps = n_steps

    def step(self):

        x = self.state
        r0 = initial_momentum(x, self.dim, self.mass_chol)
        y, r = x, r0

        for i in range(self.n_steps):
            y, r = leapfrog(y, r, self.step_size, self.model.grad, self.mass_chol)

        if accept(x, r0, y, r, self.model.logp, self.mass):
            x = y
            self._accepted += 1

        self.state = x
        self._sampled += 1
        return x

    @property
    def acceptance_rate(self):
        return self._accepted/self._sampled


def leapfrog(x, r, step_size, grad, mass_chol):
    r = r + step_size/2*grad(x)
    x = x + step_size*tostate(x, la.cho_solve((mass_chol, True), r.tovector()))
    r = r + step_size/2*grad(x)
    return x, r


def accept(x, r0, y, r, logp, mass_chol):
    A = min(0, energy(logp, y, r, mass_chol) - energy(logp, x, r0, mass_chol))
    return (np.log(np.random.rand()) < A)


def energy(logp, x, r, mass_chol):
    r = r.tovector()
    return logp(x) - 0.5*np.dot(r, la.cho_solve((mass_chol, True), r))


def initial_momentum(state, dim, mass_chol):
    r = np.dot(mass_chol, np.random.randn(dim))
    return tostate(state, r)


def tostate(ref_state, vec):
    new = State.fromkeys(ref_state.keys())
    i = 0
    for var in new:
        j = np.size(ref_state[var])
        new.update({var: vec[i:i+j].reshape(np.shape(ref_state[var]))})
        i += j
    return new


def default_mass(state, scale, condition):
    mass = np.diag(scale.tovector())
    if condition is None:
        return mass

    frozen_vars = condition
    l = []
    i = 0
    for var in state:
        j = np.size(state[var])
        if var not in frozen_vars:
            l.extend(np.arange(i, i+j))
        i += j
    return mass[np.ix_(l, l)]
