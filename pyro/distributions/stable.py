# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

from dataclasses import dataclass

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from pyro.distributions.torch_distribution import TorchDistribution


def _unsafe_standard_stable(alpha, beta, V, W, coords):
    # Implements a noisily reparametrized version of the sampler
    # Chambers-Mallows-Stuck method as corrected by Weron [1,3] and simplified
    # by Nolan [4]. This will fail if alpha is close to 1.

    # Differentiably transform noise via parameters.
    assert V.shape == W.shape
    inv_alpha = alpha.reciprocal()
    half_pi = math.pi / 2
    eps = torch.finfo(V.dtype).eps
    # make V belong to the open interval (-pi/2, pi/2)
    V = V.clamp(min=2 * eps - half_pi, max=half_pi - 2 * eps)
    ha = half_pi * alpha
    b = beta * ha.tan()
    # +/- `ha` term to keep the precision of alpha * (V + half_pi) when V ~ -half_pi
    v = b.atan() - ha + alpha * (V + half_pi)
    Z = (
        v.sin()
        / ((1 + b * b).rsqrt() * V.cos()).pow(inv_alpha)
        * ((v - V).cos().clamp(min=eps) / W).pow(inv_alpha - 1)
    )
    Z.data[Z.data != Z.data] = 0  # drop occasional NANs

    # Optionally convert to Nolan's parametrization S^0 where samples depend
    # continuously on (alpha,beta), allowing interpolation around the hole at
    # alpha=1.
    if coords == "S0":
        return Z - b
    elif coords == "S":
        return Z
    else:
        raise ValueError("Unknown coords: {}".format(coords))


RADIUS = 0.01


def _standard_stable(alpha, beta, aux_uniform, aux_exponential, coords):
    """
    Differentiably transform two random variables::

        aux_uniform ~ Uniform(-pi/2, pi/2)
        aux_exponential ~ Exponential(1)

    to a standard ``Stable(alpha, beta)`` random variable.
    """
    # Determine whether a hole workaround is needed.
    with torch.no_grad():
        hole = 1.0
        near_hole = (alpha - hole).abs() <= RADIUS
    if not torch._C._get_tracing_state() and not near_hole.any():
        return _unsafe_standard_stable(
            alpha, beta, aux_uniform, aux_exponential, coords=coords
        )
    if coords == "S":
        # S coords are discontinuous, so interpolate instead in S0 coords.
        Z = _standard_stable(alpha, beta, aux_uniform, aux_exponential, "S0")
        return torch.where(alpha == 1, Z, Z + beta * (math.pi / 2 * alpha).tan())

    # Avoid the hole at alpha=1 by interpolating between pairs
    # of points at hole-RADIUS and hole+RADIUS.
    aux_uniform_ = aux_uniform.unsqueeze(-1)
    aux_exponential_ = aux_exponential.unsqueeze(-1)
    beta_ = beta.unsqueeze(-1)
    alpha_ = alpha.unsqueeze(-1).expand(alpha.shape + (2,)).contiguous()
    with torch.no_grad():
        lower, upper = alpha_.unbind(-1)
        lower.data[near_hole] = hole - RADIUS
        upper.data[near_hole] = hole + RADIUS
        # We don't need to backprop through weights, since we've pretended
        # alpha_ is reparametrized, even though we've clamped some values.
        #               |a - a'|
        # weight = 1 - ----------
        #              2 * RADIUS
        weights = (alpha_ - alpha.unsqueeze(-1)).abs_().mul_(-1 / (2 * RADIUS)).add_(1)
        weights[~near_hole] = 0.5
    pairs = _unsafe_standard_stable(
        alpha_, beta_, aux_uniform_, aux_exponential_, coords=coords
    )
    return (pairs * weights).sum(-1)


class Stable(TorchDistribution):
    r"""
    Levy :math:`\alpha`-stable distribution. See [1] for a review.

    This uses Nolan's parametrization [2] of the ``loc`` parameter, which is
    required for continuity and differentiability. This corresponds to the
    notation :math:`S^0_\alpha(\beta,\sigma,\mu_0)` of [1], where
    :math:`\alpha` = stability, :math:`\beta` = skew, :math:`\sigma` = scale,
    and :math:`\mu_0` = loc. To instead use the S parameterization as in scipy,
    pass ``coords="S"``, but BEWARE this is discontinuous at ``stability=1``
    and has poor geometry for inference.

    This implements a reparametrized sampler :meth:`rsample` , but does not
    implement :meth:`log_prob` . Inference can be performed using either
    likelihood-free algorithms such as
    :class:`~pyro.infer.energy_distance.EnergyDistance`, or reparameterization
    via the :func:`~pyro.poutine.handlers.reparam` handler with one of the
    reparameterizers :class:`~pyro.infer.reparam.stable.LatentStableReparam` ,
    :class:`~pyro.infer.reparam.stable.SymmetricStableReparam` , or
    :class:`~pyro.infer.reparam.stable.StableReparam` e.g.::

        with poutine.reparam(config={"x": StableReparam()}):
            pyro.sample("x", Stable(stability, skew, scale, loc))

    or simply wrap in :class:`~pyro.infer.reparam.strategies.MinimalReparam` or
    :class:`~pyro.infer.reparam.strategies.AutoReparam` , e.g.::

        @MinimalReparam()
        def model():
            ...

    [1] S. Borak, W. Hardle, R. Weron (2005).
        Stable distributions.
        https://edoc.hu-berlin.de/bitstream/handle/18452/4526/8.pdf
    [2] J.P. Nolan (1997).
        Numerical calculation of stable densities and distribution functions.
    [3] Rafal Weron (1996).
        On the Chambers-Mallows-Stuck Method for
        Simulating Skewed Stable Random Variables.
    [4] J.P. Nolan (2017).
        Stable Distributions: Models for Heavy Tailed Data.
        https://edspace.american.edu/jpnolan/wp-content/uploads/sites/1720/2020/09/Chap1.pdf

    :param Tensor stability: Levy stability parameter :math:`\alpha\in(0,2]` .
    :param Tensor skew: Skewness :math:`\beta\in[-1,1]` .
    :param Tensor scale: Scale :math:`\sigma > 0` . Defaults to 1.
    :param Tensor loc: Location :math:`\mu_0` when using Nolan's S0
        parametrization [2], or :math:`\mu` when using the S parameterization.
        Defaults to 0.
    :param str coords: Either "S0" (default) to use Nolan's continuous S0
        parametrization, or "S" to use the discontinuous parameterization.
    """

    has_rsample = True
    arg_constraints = {
        "stability": constraints.interval(0, 2),  # half-open (0, 2]
        "skew": constraints.interval(-1, 1),  # closed [-1, 1]
        "scale": constraints.positive,
        "loc": constraints.real,
    }
    support = constraints.real

    def __init__(
        self, stability, skew, scale=1.0, loc=0.0, coords="S0", validate_args=None
    ):
        assert coords in ("S", "S0"), coords
        self.stability, self.skew, self.scale, self.loc = broadcast_all(
            stability, skew, scale, loc
        )
        self.coords = coords
        super().__init__(self.loc.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Stable, _instance)
        batch_shape = torch.Size(batch_shape)
        for name in self.arg_constraints:
            setattr(new, name, getattr(self, name).expand(batch_shape))
        new.coords = self.coords
        super(Stable, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        # Auxiliary calculations
        aux = _unsafe_stable_given_uniform_aux_calc(self.stability, self.skew)

        # Undo shift and scale
        value = (value - self.loc) / self.scale
        
        N = 500

        log_prob = -math.inf * torch.ones(value.shape)
        for n, (idx, bias) in enumerate(
                [(value < aux.shift, -0.5 * math.pi * torch.ones(aux.u_zero.shape)),
                 (torch.logical_xor(value > aux.shift, aux.u_zero > 0), aux.u_min),
                 (value > aux.shift, aux.u_max)]):
            u = torch.arange(1, N + 1) / (N + 1)
            u = u[[None] * len(value[idx].shape) + [slice(None)]]
            alpha, beta, u, v, probs, bias = broadcast_all(self.stability[idx][..., None],
                                                           self.skew[idx][..., None], u,
                                                           value[idx][..., None],
                                                           aux.probs[..., n][idx][..., None],
                                                           bias[idx][..., None])
            u = (probs * u) * math.pi + bias
            partial_log_prob, W = _unsafe_stable_given_uniform_logprob(alpha, beta, u, v, self.coords)
            partial_log_prob = torch.logsumexp(partial_log_prob, dim = -1) + aux.probs[..., n][idx].log()
            log_prob[idx] = torch.logsumexp(torch.stack((log_prob[idx],
                                                         partial_log_prob), dim=0), dim=0)

        return log_prob - self.scale.log() - torch.tensor(N).log()

    def rsample(self, sample_shape=torch.Size()):
        # Draw parameter-free noise.
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            new_empty = self.stability.new_empty
            aux_uniform = new_empty(shape).uniform_(-math.pi / 2, math.pi / 2)
            aux_exponential = new_empty(shape).exponential_()

        # Differentiably transform.
        x = _standard_stable(
            self.stability, self.skew, aux_uniform, aux_exponential, coords=self.coords
        )
        return self.loc + self.scale * x

    @property
    def mean(self):
        result = self.loc
        if self.coords == "S0":
            result = (
                result - self.scale * self.skew * (math.pi / 2 * self.stability).tan()
            )
        return result.masked_fill(self.stability <= 1, math.nan)

    @property
    def variance(self):
        var = self.scale * self.scale
        return var.mul(2).masked_fill(self.stability < 2, math.inf)


class StableGivenUniform(TorchDistribution):
    r"""
    Stable distribution given the value of the uniformaly distributed reparametrized
    auxiliary variable of the CMS procedure.

    :param Tensor stability: Levy stability parameter :math:`\alpha\in(0,2]` .
    :param Tensor skew: Skewness :math:`\beta\in[-1,1]` .
    :param Tensor scale: Scale :math:`\sigma > 0` . Defaults to 1.
    :param Tensor loc: Location :math:`\mu_0` when using Nolan's S0
        parametrization [2], or :math:`\mu` when using the S parameterization.
        Defaults to 0.
    :param: Tensor u_low: Uniformaly distributed auxiliary variable of the
        modified CMS procedure :math:`U \in (0,1)` .
    :param: Tensor u_mid: Uniformaly distributed auxiliary variable of the
        modified CMS procedure :math:`U \in (0,1)` .
    :param: Tensor u_high: Uniformaly distributed auxiliary variable of the
        modified CMS procedure :math:`U \in (0,1)` .
    :param str coords: Either "S0" (default) to use Nolan's continuous S0
        parametrization, or "S" to use the discontinuous parameterization.
    """

    has_rsample = False
    arg_constraints = {
        "stability": constraints.interval(0, 2),  # half-open (0, 2]
        "skew": constraints.interval(-1, 1),  # closed [-1, 1]
        "scale": constraints.positive,
        "loc": constraints.real,
        "u_low": constraints.interval(0, 1),
        "u_mid": constraints.interval(0, 1),
        "u_high": constraints.interval(0, 1)
    }
    support = constraints.real

    def __init__(
        self, stability, skew, scale=1.0, loc=0.0, u_low=0.0, u_mid=0.0, u_high=0.0, coords="S0", validate_args=None
    ):
        assert coords in ("S", "S0"), coords
        self.stability, self.skew, self.scale, self.loc, self.u_low, self.u_mid, self.u_high = broadcast_all(
            stability, skew, scale, loc, u_low, u_mid, u_high
        )
        self.coords = coords
        super().__init__(self.loc.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Stable, _instance)
        batch_shape = torch.Size(batch_shape)
        for name in self.arg_constraints:
            setattr(new, name, getattr(self, name).expand(batch_shape))
        new.coords = self.coords
        super(Stable, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        # Auxiliary calculations
        aux = _unsafe_stable_given_uniform_aux_calc(self.stability, self.skew)

        # Undo shift and scale
        value = (value - self.loc) / self.scale
        
        log_prob = -math.inf * torch.ones(value.shape)
        for n, (idx, bias, u) in enumerate(
                [(value < aux.shift, -0.5 * math.pi * torch.ones(aux.u_zero.shape), self.u_low),
                 (torch.logical_xor(value > aux.shift, aux.u_zero > 0), aux.u_min, self.u_mid),
                 (value > aux.shift, aux.u_max, self.u_high)]):
            u = (aux.probs[..., n] * u)[idx] * math.pi + bias[idx]
            partial_log_prob, W = _unsafe_stable_given_uniform_logprob(self.stability[idx], self.skew[idx], u, value[idx], self.coords)
            partial_log_prob = partial_log_prob + aux.probs[..., n][idx].log()
            log_prob[idx] = torch.logsumexp(torch.stack((log_prob[idx],
                                                         partial_log_prob), dim=0), dim=0)

        if torch.isnan(log_prob).any():
            raise

        return log_prob - self.scale.log()

    def sample(self, sample_shape=torch.Size()):
        # Draw parameter-free noise.
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            aux_exponential = self.stability.new_empty(shape).exponential_()

        # Match dimensions of preset noise and drawn noise.
        u_low, u_mid, u_high, aux_exponential = broadcast_all(self.u_low, self.u_mid, self.u_high, aux_exponential)

        # Auxiliary calculations
        aux = _unsafe_stable_given_uniform_aux_calc(self.stability, self.skew)
        
        # Select range for auxiliary variable
        interval = torch.distributions.Categorical(aux.probs).sample()

        aux_uniform = torch.zeros(aux_exponential.shape)
        for n, (bias, u) in enumerate([(-0.5 * math.pi * torch.ones(aux.u_zero.shape), u_low),
                                       (aux.u_min, u_mid),
                                       (aux.u_max, u_high)]):
            idx = interval == n
            aux_uniform[idx] = (aux.probs[..., n] * u)[idx] * math.pi + bias[idx]
            
        # Differentiably transform.
        x = _standard_stable(
            self.stability, self.skew, aux_uniform, aux_exponential, coords=self.coords
        )
        return self.loc + self.scale * x


@dataclass(frozen=True, eq=False)
class _unsafe_stable_given_uniform_aux_calc_results():
    u_zero: torch.Tensor
    u_zero_complement: torch.Tensor
    shift: torch.Tensor
    u_min: torch.Tensor
    u_max: torch.Tensor
    probs: torch.Tensor


def _unsafe_stable_given_uniform_aux_calc(alpha, beta):
    ha = math.pi / 2 * alpha
    b = beta * ha.tan()
    atan_b = b.atan()
    shift = -b
    u_zero = -alpha.reciprocal() * atan_b
    u_zero_complement = (2 - alpha).reciprocal() * atan_b
    u_all = torch.stack((u_zero, u_zero_complement), dim=0)
    u_min = u_all.min(dim=0).values
    u_max = u_all.max(dim=0).values
    probs = torch.stack((0.5 + u_min / math.pi,
                            (u_max - u_min) / math.pi,
                            0.5 - u_max / math.pi), dim=-1)
    return _unsafe_stable_given_uniform_aux_calc_results(u_zero, u_zero_complement, shift,
                                                         u_min, u_max, probs)


def _unsafe_stable_given_uniform_logprob(alpha, beta, V, Z, coords):
    # Calculate log-probability of Z given V. This will fail if alpha is close to 1.

    # Differentiably transform noise via parameters.
    assert V.shape == Z.shape
    inv_alpha = alpha.reciprocal()
    half_pi = math.pi / 2
    eps = torch.finfo(V.dtype).eps
    # make V belong to the open interval (-pi/2, pi/2)
    V = V.clamp(min=2 * eps - half_pi, max=half_pi - 2 * eps)
    ha = half_pi * alpha
    b = beta * ha.tan()

    # Optionally convert from Nolan's parametrization S^0 where samples depend
    # continuously on (alpha,beta), allowing interpolation around the hole at
    # alpha=1.
    if coords == "S0":
        Z = Z + b
    elif coords != "S":
        raise ValueError("Unknown coords: {}".format(coords))
    
    # +/- `ha` term to keep the precision of alpha * (V + half_pi) when V ~ -half_pi
    v = b.atan() - ha + alpha * (V + half_pi)
    W = ( ( v.sin() / Z /
           ((1 + b * b).rsqrt() * V.cos()).pow(inv_alpha)
          ).pow(alpha / (1 - alpha))
          * (v - V).cos().clamp(min=eps) 
        )
    
    return -W + (alpha * W / Z / (alpha - 1)).abs().log(), W
    
