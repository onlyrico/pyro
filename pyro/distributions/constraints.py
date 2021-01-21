# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions.constraints import *  # noqa F403
from torch.distributions.constraints import Constraint
from torch.distributions.constraints import __all__ as torch_constraints
from torch.distributions.constraints import lower_cholesky

import pyro.distributions.torch_patch  # noqa F403


# backport of https://github.com/pytorch/pytorch/pull/50547
def _():
    module = torch.distributions.constraints

    static_is_discrete = {
        "Boolean": True,
        "Constraint": False,
        "IntegerGreaterThan": True,
        "IntegerInterval": True,
        "Multinomial": True,
        "OneHot": True,
    }
    for name, is_discrete in static_is_discrete.items():
        cls = getattr(module, "_" + name, None)  # Old private names.
        cls = getattr(module, name, cls)  # New public names.
        if cls is not None:  # Ignore PyTorch version mismatch.
            cls.is_discrete = is_discrete

    static_event_dim = {
        "Constraint": 0,
        "Simplex": 1,
        "Multinomial": 1,
        "OneHot": 1,
        "LowerTriangular": 2,
        "LowerCholesky": 2,
        "CorrCholesky": 2,
        "PositiveDefinite": 2,
        "RealVector": 1,
    }
    for name, event_dim in static_event_dim.items():
        cls = getattr(module, "_" + name, None)  # Old private names.
        cls = getattr(module, name, cls)  # New public names.
        if cls is not None:  # Ignore PyTorch version mismatch.
            cls.event_dim = event_dim


_()


# TODO remove after https://github.com/pytorch/pytorch/pull/50547
class IndependentConstraint(Constraint):
    """
    Wraps a constraint by aggregating over ``reinterpreted_batch_ndims``-many
    dims in :meth:`check`, so that an event is valid only if all its
    independent entries are valid.

    :param torch.distributions.constraints.Constraint base_constraint: A base
        constraint whose entries are incidentally independent.
    :param int reinterpreted_batch_ndims: The number of extra event dimensions that will
        be considered dependent.
    """
    def __init__(self, base_constraint, reinterpreted_batch_ndims):
        self.base_constraint = base_constraint
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    @property
    def is_discrete(self):
        return self.base_constraint.is_discrete

    @property
    def event_dim(self):
        return self.base_constraint.event_dim + self.reinterpreted_batch_ndims

    def check(self, value):
        result = self.base_constraint.check(value)
        result = result.reshape(result.shape[:result.dim() - self.reinterpreted_batch_ndims] + (-1,))
        result = result.all(-1)
        return result


# backport of https://github.com/pytorch/pytorch/pull/50547
class _Multinomial(Constraint):
    """
    Constrain to nonnegative integer values summing to at most an upper bound.

    Note due to limitations of the Multinomial distribution, this currently
    checks the weaker condition ``value.sum(-1) <= upper_bound``. In the future
    this may be strengthened to ``value.sum(-1) == upper_bound``.
    """
    is_discrete = True
    event_dim = 1

    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def check(self, x):
        return (x >= 0).all(dim=-1) & (x.sum(dim=-1) <= self.upper_bound)


# TODO move this upstream to torch.distributions
class _Integer(Constraint):
    """
    Constrain to integers.
    """
    is_discrete = True

    def check(self, value):
        return value % 1 == 0

    def __repr__(self):
        return self.__class__.__name__[1:]


class _Sphere(Constraint):
    """
    Constrain to the Euclidean sphere of any dimension.
    """
    event_dim = 1
    reltol = 10.  # Relative to finfo.eps.

    def check(self, value):
        eps = torch.finfo(value.dtype).eps
        try:
            norm = torch.linalg.norm(value, dim=-1)  # torch 1.7+
        except AttributeError:
            norm = value.norm(dim=-1)  # torch 1.6
        error = (norm - 1).abs()
        return error < self.reltol * eps * value.size(-1) ** 0.5

    def __repr__(self):
        return self.__class__.__name__[1:]


class _CorrCholesky(Constraint):
    """
    Constrains to lower-triangular square matrices with positive diagonals and
    Euclidean norm of each row is 1, such that `torch.mm(omega, omega.t())` will
    have unit diagonal.
    """
    event_dim = 2

    def check(self, value):
        unit_norm_row = (value.norm(dim=-1).sub(1) < 1e-4).min(-1)[0]
        return lower_cholesky.check(value) & unit_norm_row


class _OrderedVector(Constraint):
    """
    Constrains to a real-valued tensor where the elements are monotonically
    increasing along the `event_shape` dimension.
    """
    event_dim = 1

    def check(self, value):
        if value.ndim == 0:
            return torch.tensor(False, device=value.device)
        elif value.shape[-1] == 1:
            return torch.ones_like(value[..., 0], dtype=bool)
        else:
            return torch.all(value[..., 1:] > value[..., :-1], dim=-1)


corr_cholesky_constraint = _CorrCholesky()
independent = IndependentConstraint
integer = _Integer()
multinomial = _Multinomial
ordered_vector = _OrderedVector()
sphere = _Sphere()

__all__ = [
    'IndependentConstraint',
    'corr_cholesky_constraint',
    'independent',
    'integer',
    'multinomial',
    'ordered_vector',
    'sphere',
]

__all__.extend(torch_constraints)
__all__ = sorted(set(__all__))
del torch_constraints


# Create sphinx documentation.
__doc__ = """
    Pyro's constraints library extends
    :mod:`torch.distributions.constraints`.
"""
__doc__ += "\n".join([
    """
    {}
    ----------------------------------------------------------------
    {}
    """.format(
        _name,
        "alias of :class:`torch.distributions.constraints.{}`".format(_name)
        if globals()[_name].__module__.startswith("torch") else
        ".. autoclass:: {}".format(_name if type(globals()[_name]) is type else
                                   type(globals()[_name]).__name__)
    )
    for _name in sorted(__all__)
])
