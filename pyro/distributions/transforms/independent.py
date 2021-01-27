# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from torch.distributions.transforms import Transform

from .. import constraints
from ..util import sum_rightmost


# backport of https://github.com/pytorch/pytorch/pull/50581
class IndependentTransform(Transform):
    def __init__(self, base_transform, reinterpreted_batch_ndims, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.base_transform = base_transform.with_cache(cache_size)
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return IndependentTransform(self.base_transform,
                                    self.reinterpreted_batch_ndims,
                                    cache_size=cache_size)

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        return constraints.independent(self.base_transform.domain,
                                       self.reinterpreted_batch_ndims)

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        return constraints.independent(self.base_transform.codomain,
                                       self.reinterpreted_batch_ndims)

    @property
    def bijective(self):
        return self.base_transform.bijective

    @property
    def sign(self):
        return self.base_transform.sign

    def _call(self, x):
        if x.dim() < self.domain.event_dim:
            raise ValueError("Too few dimensions on input")
        return self.base_transform(x)

    def _inverse(self, y):
        if y.dim() < self.codomain.event_dim:
            raise ValueError("Too few dimensions on input")
        return self.base_transform.inv(y)

    def log_abs_det_jacobian(self, x, y):
        result = self.base_transform.log_abs_det_jacobian(x, y)
        result = sum_rightmost(result, self.reinterpreted_batch_ndims)
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.base_transform)}, {self.reinterpreted_batch_ndims})"

    def forward_shape(self, shape):
        return self.base_transform.forward_shape(shape)

    def inverse_shape(self, shape):
        return self.base_transform.inverse_shape(shape)
