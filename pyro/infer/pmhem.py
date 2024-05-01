# SPDX-License-Identifier: Apache-2.0

import torch

import pyro.poutine as poutine
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import is_validation_enabled
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_if_enumerated, check_model_guide_match, warn_if_nan


class PMHEM(ELBO):
    r"""
    An implementation of particle Metropolis-Hastings evidence maximization.

    The objective is to simultaneously:

    -   Maximize the ``model`` evidence.
    -   Reduce the variance of the evidence estimator.

    The variance of the evidence estimator is depednent on how well the ``guide`` is tracking the ``model``.
    In case the ``model`` has no parameters only the variance of the evidence estimator is reduced via matching
    the ``guide`` to the ``model``.

    :param int num_particles: The number of particles/samples used to form the objective
        (gradient) estimator. Default is 1.
    :param bool model_has_params: Indicate if model has learnable params. Useful in avoiding extra
        computation. Default is True.
    :param bool vectorize_particles: Whether the traces should be vectorised
        across `num_particles`. Default is True.
    :param int max_plate_nesting: Bound on max number of nested
        :func:`pyro.plate` contexts. Default is infinity.
    :param int num_updates: Number of target distribution updates. Default is 1.
    """

    def __init__(
        self,
        num_particles=1,
        model_has_params=True,
        vectorize_particles=True,
        max_plate_nesting=float("inf"),
        num_updates=1):
        super().__init__(
            num_particles=num_particles,
            max_plate_nesting=max_plate_nesting,
            vectorize_particles=vectorize_particles)
        self.model_has_params = model_has_params
        self.log_weights = None
        self.num_updates = num_updates

    def _get_trace(self, model, guide, args, kwargs):
        """
        Returns a single trace from the guide, and the model that is run against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, args, kwargs, detach=True
        )
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace

    def _get_log_prob(self, trace):
        """
        Calculate log-probability of a trace.
        """
        log_prob = 0.0

        for _, site in trace.nodes.items():
            if site["type"] == "sample":
                if self.vectorize_particles:
                    log_prob_site = (
                        site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                    )
                else:
                    log_prob_site = site["log_prob_sum"]
                log_prob = log_prob + log_prob_site

        return log_prob

    def _get_traces_and_weights(self, model, guide, args, kwargs):
        """
        Get model and guide traces and calculate their probabilities.
        """
        model_traces, guide_traces = zip(*self._get_traces(model, guide, args, kwargs))

        log_weights = []

        for model_trace, guide_trace in zip(model_traces, guide_traces):
            log_p = self._get_log_prob(model_trace)
            log_q = self._get_log_prob(guide_trace)
            log_weights.append(log_p - log_q)

        return model_traces, guide_traces, log_weights

    def _update_traces(self, model, guide, args, kwargs):
        _, new_traces, new_log_weights = \
                        self._get_traces_and_weights(model, guide, args, kwargs)
        
        for n, (trace, log_weight, new_trace, new_log_weight) in \
                enumerate(zip(self.traces, self.log_weights, new_traces, new_log_weights)):
            if log_weight.numel() <= 1:
                if torch.rand() <= (new_log_weight - log_weight).clamp(max=0).exp():
                    self.traces[n] = new_trace
                    self.log_weights[n] = new_log_weight
            else:
                prob = torch.clamp(new_log_weight - log_weight, max=0.0).exp()
                idx = torch.rand(*prob.shape) <= prob
                for name, site in trace.nodes.items():
                    if site["type"] == "sample":
                        site["value"][idx] = new_trace.nodes[name]["value"][idx]
                self.log_weights[n][idx] = new_log_weight[idx]

    def _loss(self, model, guide, args, kwargs):
        """
        :returns: returns model loss and guide loss
        :rtype: float, float
        """

        with torch.no_grad():
            if self.log_weights is None:
                _, self.traces, self.log_weights = \
                            self._get_traces_and_weights(model, guide, args, kwargs)
                
            for _ in range(self.num_updates):
                self._update_traces(model, guide, args, kwargs)

        model_log_prob = torch.tensor(0.0)
        guide_log_prob = torch.tensor(0.0)

        for trace in self.traces:
            if self.model_has_params:
                model_trace = poutine.trace(poutine.replay(model, trace=trace),
                                            graph_type="flat").get_trace(*args, **kwargs)
                model_trace = prune_subsample_sites(model_trace)
                model_trace.compute_log_prob()
                model_log_prob = model_log_prob - self._get_log_prob(model_trace).sum()

            guide_trace = poutine.trace(poutine.replay(guide, trace=trace),
                                        graph_type="flat").get_trace(*args, **kwargs)
            guide_trace = prune_subsample_sites(guide_trace)
            guide_trace.compute_score_parts()
            guide_log_prob = guide_log_prob - self._get_log_prob(guide_trace).sum()

        return (model_log_prob / self.num_particles,
                guide_log_prob / self.num_particles)

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns model loss and guide loss
        :rtype: float, float
        """
        with torch.no_grad():
            model_log_prob, guide_log_prob = self._loss(model, guide, args, kwargs)

        return model_log_prob, guide_log_prob

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns model loss and guide loss
        :rtype: float
        """
        model_log_prob, guide_log_prob = self._loss(model, guide, args, kwargs)
        # convenience addition to ensure easier gradients without requiring `retain_graph=True`
        (model_log_prob + guide_log_prob).backward()

        return model_log_prob.detach().item(), guide_log_prob.detach().item()
