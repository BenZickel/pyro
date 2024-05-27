# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from functools import partial

import torch

value_near_zero_tolerance = 0.01
alpha_near_one_tolerance = 0.05


finfo = torch.finfo(torch.float64)
MAX_LOG = math.log10(finfo.max)
MIN_LOG = math.log10(finfo.tiny)


def create_integrator(num_points):
    from scipy.special import roots_legendre

    roots, weights = roots_legendre(num_points)
    roots = torch.Tensor(roots).double()
    weights = torch.Tensor(weights).double()
    log_weights = weights.log()
    half_roots = roots * 0.5

    def integrate(fn, domain):
        sl = [slice(None)] + (len(domain.shape) - 1) * [None]
        half_roots_sl = half_roots[sl]
        value = domain[0] * (0.5 - half_roots_sl) + domain[1] * (0.5 + half_roots_sl)
        return (
            torch.logsumexp(fn(value) + log_weights[sl], dim=0)
            + ((domain[1] - domain[0]) / 2).log()
        )

    return integrate


def set_integrator(num_points):
    global integrate
    integrate = create_integrator(num_points)


# Stub which is replaced by the default integrator when called for the first time
# if a default integrator has not already been set.
def integrate(*args, **kwargs):
    set_integrator(num_points=501)
    return integrate(*args, **kwargs)


class StableLogProb:
    def log_prob(self, value):
        # Undo shift and scale
        value = (value - self.loc) / self.scale

        # Use double precision math
        alpha = self.stability.double()
        beta = self.skew.double()
        value = value.double()

        return _stable_log_prob(alpha, beta, value, self.coords) - self.scale.log()


def _stable_log_prob(alpha, beta, value, coords):
    # Convert to Nolan's parametrization S^0 where samples depend
    # continuously on (alpha,beta), allowing interpolation around the hole at
    # alpha=1.
    if coords == "S":
        value = torch.where(
            alpha == 1, value, value - beta * (math.pi / 2 * alpha)
        ).tan()
    elif coords != "S0":
        raise ValueError("Unknown coords: {}".format(coords))

    # Find near one alpha
    idx = (alpha - 1).abs() < alpha_near_one_tolerance

    log_prob = _unsafe_alpha_stable_log_prob_S0(
        torch.where(idx, 1 + alpha_near_one_tolerance, alpha), beta, value
    )

    # Handle alpha near one by interpolation
    if idx.any():
        log_prob_pos = log_prob[idx]
        log_prob_neg = _unsafe_alpha_stable_log_prob_S0(
            (1 - alpha_near_one_tolerance) * log_prob_pos.new_ones(log_prob_pos.shape),
            beta[idx],
            value[idx],
        )
        weights = (alpha[idx] - 1) / (2 * alpha_near_one_tolerance) + 0.5
        log_prob[idx] = torch.logsumexp(
            torch.stack(
                (log_prob_pos + weights.log(), log_prob_neg + (1 - weights).log()),
                dim=0,
            ),
            dim=0,
        )

    return log_prob


def _unsafe_alpha_stable_log_prob_S0(alpha, beta, Z0):
    # Calculate log-probability of Z0 in Nolan's parametrization S^0. This will fail if alpha is close to 1

    # Convert from Nolan's parametrization S^0 where samples depend
    # continuously on (alpha,beta), allowing interpolation around the hole at
    # alpha=1.
    shift = beta * (math.pi / 2 * alpha).tan()
    Z = Z0 + shift

    # Find near zero values
    zero_value_log_prob = _unsafe_alpha_stable_log_prob_at_zero(alpha, beta)
    per_alpha_value_near_zero_tolerance = (
        value_near_zero_tolerance * zero_value_log_prob.exp().reciprocal()
    )
    idx = Z.abs() < per_alpha_value_near_zero_tolerance

    # Calculate log-prob at safe values
    log_prob = _unsafe_stable_log_prob(
        alpha, beta, torch.where(idx, per_alpha_value_near_zero_tolerance, Z)
    )

    # Handle near zero values by interpolation
    if idx.any():
        log_prob_pos = log_prob[idx]
        log_prob_zero = zero_value_log_prob[idx]
        log_prob_neg = _unsafe_stable_log_prob(
            alpha[idx], beta[idx], -per_alpha_value_near_zero_tolerance[idx]
        )
        p1, p2 = _natural_cubic_spline_minus_one_zero_plus_one(
            log_prob_neg,
            log_prob_zero,
            log_prob_pos,
            Z[idx] / per_alpha_value_near_zero_tolerance[idx],
        )
        log_prob[idx] = torch.where(Z[idx] < 0, p1, p2)

    return log_prob


def _unsafe_stable_log_prob(alpha, beta, Z):
    # Calculate log-probability of Z. This will fail if alpha is close to 1
    # or if Z is close to 0
    ha = math.pi / 2 * alpha
    b = beta * ha.tan()
    atan_b = b.atan()
    u_zero = -alpha.reciprocal() * atan_b

    # If sample should be negative calculate with flipped beta and flipped value
    flip_beta_x = Z < 0
    beta = torch.where(flip_beta_x, -beta, beta)
    u_zero = torch.where(flip_beta_x, -u_zero, u_zero)
    Z = torch.where(flip_beta_x, -Z, Z)

    # Set integration domwin
    domain = torch.stack((u_zero, 0.5 * math.pi * u_zero.new_ones(u_zero.shape)), dim=0)

    integrand = partial(
        _unsafe_stable_given_uniform_log_prob, alpha=alpha, beta=beta, Z=Z
    )

    return integrate(integrand, domain) - math.log(math.pi)


def _unsafe_stable_given_uniform_log_prob(V, alpha, beta, Z):
    # Calculate log-probability of Z given V. This will fail if alpha is close to 1
    # or if Z is close to 0
    inv_alpha_minus_one = (alpha - 1).reciprocal()
    half_pi = math.pi / 2
    eps = torch.finfo(V.dtype).eps
    # make V belong to the open interval (-pi/2, pi/2)
    V = V.clamp(min=2 * eps - half_pi, max=half_pi - 2 * eps)
    ha = half_pi * alpha
    b = beta * ha.tan()
    atan_b = b.atan()
    cos_V = V.cos()

    # +/- `ha` term to keep the precision of alpha * (V + half_pi) when V ~ -half_pi
    v = atan_b - ha + alpha * (V + half_pi)

    term1_log = atan_b.cos().log() * inv_alpha_minus_one
    term2_log = (Z * cos_V / v.sin()).log() * alpha * inv_alpha_minus_one
    term3_log = ((v - V).cos() / cos_V).log()

    W_log = term1_log + term2_log + term3_log

    W = W_log.clamp(min=MIN_LOG, max=MAX_LOG).exp()

    log_prob = -W + (alpha * W / Z / (alpha - 1)).abs().log()

    # Infinite W means zero-probability
    log_prob = torch.where(W == torch.inf, -torch.inf, log_prob)

    log_prob = log_prob.clamp(min=MIN_LOG, max=MAX_LOG)

    return log_prob


def _unsafe_alpha_stable_log_prob_at_zero(alpha, beta):
    # Calculate log-probability at value of zero. This will fail if alpha is close to 1
    inv_alpha = alpha.reciprocal()
    half_pi = math.pi / 2
    ha = half_pi * alpha
    b = beta * ha.tan()
    atan_b = b.atan()

    term1_log = (inv_alpha * atan_b).cos().log()
    term2_log = atan_b.cos().log() * inv_alpha
    term3_log = torch.lgamma(1 + inv_alpha)

    log_prob = term1_log - term2_log + term3_log - math.log(math.pi)

    log_prob = log_prob.clamp(min=MIN_LOG, max=MAX_LOG)

    return log_prob


def _natural_cubic_spline_minus_one_zero_plus_one(y1, y2, y3, x):
    sum_y = y1 + y3 - 2 * y2
    aux_y = y3 - y2
    a = -0.75 * sum_y + 1.5 * aux_y
    b = 1.5 * sum_y
    c = 1.25 * sum_y - 0.5 * aux_y
    f = -0.75 * sum_y - 0.5 * aux_y
    common = y2 + a * x + b * x**2
    return common + c * x**3, common + f * x**3