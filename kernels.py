from dataclasses import dataclass

from beartype.typing import Union
import jax.numpy as jnp
from jaxtyping import Float
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from gpjax.base import param_field
from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.stationary.utils import squared_distance
from gpjax.typing import (
    Array,
    ScalarFloat,
)

@dataclass
class OrnsteinUhlenbeck(AbstractKernel):
    r"""The Ornstein-Uhlenbeck kernel."""

    lengthscale: Union[ScalarFloat, Float[Array, " D"]] = param_field(
        jnp.array(1.0), bijector=tfb.Softplus()
    )
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    name: str = "OU"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        r"""Compute the OU kernel between a pair of arrays.

        Evaluate the kernel on a pair of inputs $`(x, y)`$ with lengthscale parameter
        $`\ell`$ and variance $`\sigma^2`$:
        ```math
        k(x,y)=\sigma^2\exp\Bigg(- \frac{\lVert x - y \rVert_2}{2 \ell^2} \Bigg)
        ```

        Args:
            x (Float[Array, " D"]): The left hand argument of the kernel function's call.
            y (Float[Array, " D"]): The right hand argument of the kernel function's call.

        Returns:
            ScalarFloat: The value of $`k(x, y)`$.
        """
        x = self.slice_input(x) / self.lengthscale
        y = self.slice_input(y) / self.lengthscale
        K = self.variance * jnp.exp(-0.5 * jnp.sum(jnp.abs(x - y)))
        return K.squeeze()