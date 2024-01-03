import gpjax as gpx
import jax
from jax import jit
import jax.numpy as jnp
from beartype.typing import Callable
from blackjax.progress_bar import progress_bar_scan
from blackjax.base import SamplingAlgorithm
from gpjax.typing import Array, ScalarFloat
from jax.tree_util import tree_leaves
from blackjax.types import ArrayLikeTree
from beartype.typing import NamedTuple

from kernelsearch import jit_set_trainables


def log_likelihood_function(
    gp_posterior: gpx.gps.AbstractPosterior,
    lc_model: Callable,
    X: Array,
    y: Array,
    mask: Array,
    fix_gp: bool = False,
    negative: bool = False,
    compile: bool = True,
) -> Callable:
    """Create objective function for lightcurve fit.
    The probability is calculated in the following way:
    - Fit GP outside of transit (masked input)
    - Remove model lightcurve
    - Calculate probability of remaining data under
      fitted GP

    The returned Callable calculates the log probability
    for an input dict 'params', which must be a dictonary
    with the keys 'gp_parameter' and 'lc_parameter', and
    the values being arrays of the parameter.

    Parameters
    ----------
    gp_posterior : gpx.gps.AbstractPosterior
        The gaussian process to use for
        background.
    lc_model : Callable
        Model that calculates the light curve.
    X : Array
        X data, usually time.
    y : Array
        y data, usually the light curve/flux.
    mask : Array
        Boolean mask that covers the transits.
    fix_gp : bool, optional
        Whether to fix the parameter of the GP to initial
        values. In that case, the 'gp_parameter' key in
        the params input has no effect, by default False
    negative : bool, optional
        If True, return negative of probability. By
        default False
    compile : bool, optional
        Whether to compile the objective using jit, by
        default True
    Returns
    -------
    Callable
        The callable objective function. Takes 'params' as
        input, which must be a dict.
    """
    constant = jnp.array(-1.0) if negative else jnp.array(1.0)

    # D = gpx.Dataset(
    #     X=X.reshape(-1, 1),
    #     y=y.reshape(-1, 1),
    # )
    D_masked = gpx.Dataset(
        X=X[mask].reshape(-1, 1),
        y=y[mask].reshape(-1, 1),
    )
    D_inverse_masked = gpx.Dataset(
        X=X[~mask].reshape(-1, 1),
        y=y[~mask].reshape(-1, 1),
    )

    # indices of trainables for GP
    trainable_idx = jnp.argwhere(
        jnp.array(tree_leaves(gp_posterior.trainables()))
    ).reshape(-1)

    # fix gp variables to to initial values
    if fix_gp:
        # constrain the posterior
        updated_posterior = gp_posterior.constrain()

        # calculate distribution of values predicted for all x values,
        # under the masked observations
        latent_dist = updated_posterior(
            D_inverse_masked.X,
            train_data=D_masked,
        )
        predictive_dist = updated_posterior.likelihood(
            latent_dist
        )  # adds observational uncertainty

        def objective(params: Array) -> ScalarFloat:
            # calculate lightcurve model
            lightcurve = lc_model(D_inverse_masked.X, params["lc_parameter"])

            # remove lightcurve from observations and
            # calculate probability under GP model
            res = (D_inverse_masked.y - lightcurve).reshape(-1)
            log_prob = predictive_dist.log_prob(res)

            # return (negative of, if wanted) log probability
            return constant * jnp.nan_to_num(log_prob, nan=-jnp.inf)

    # adapt gp parameter at every step
    else:

        def objective(params: Array) -> ScalarFloat:
            # update the parameter of the posterior object
            updated_posterior = jit_set_trainables(
                gp_posterior,
                jnp.array(params["gp_parameter"]),
                trainable_idx,
            ).constrain()

            # calculate distribution of values
            # predicted for all x values, under
            # the masked observations
            latent_dist = updated_posterior(
                D_inverse_masked.X,
                train_data=D_masked,
            )
            predictive_dist = updated_posterior.likelihood(
                latent_dist
            )  # adds observational uncertainty

            # calculate lightcurve model
            lightcurve = lc_model(D_inverse_masked.X, params["lc_parameter"])

            # remove lightcurve from observations and
            # calculate probability under GP model
            res = (D_inverse_masked.y - lightcurve).reshape(-1)
            log_prob = predictive_dist.log_prob(res)

            # return (negative of, if wanted) log probability
            return constant * jnp.nan_to_num(log_prob, nan=-jnp.inf)

    if compile:
        return jit(objective)
    return objective


def run_inference_algorithm(
    rng_key: Array,
    initial_state_or_position: ArrayLikeTree | NamedTuple,
    inference_algorithm: SamplingAlgorithm,
    num_steps: int,
    progress_bar: bool = False,
) -> tuple:
    """The inference loop for the Blackjax sampling,
    adapted from the Blackjax code.

    Parameters
    ----------
    rng_key : Array
        jax random key
    initial_state_or_position : ArrayLikeTree | NamedTuple
        The initial position for the sampling.
    inference_algorithm : SamplingAlgorithm
        The Blackjax algorithm used for sampling.
    num_steps : int
        The number of steps for which to sample
    progress_bar : bool, optional
        Wether the progressbar should be shown, by
        default False

    Returns
    -------
    tuple
        Tuple of final_state, state_history, and
        info_history
    """
    init_key, sample_key = jax.random.split(rng_key, 2)
    try:
        initial_state = inference_algorithm.init(
            initial_state_or_position,
            init_key,  # type: ignore
        )
    except (TypeError, ValueError, AttributeError):
        # We assume initial_state is already in the right format.
        initial_state = initial_state_or_position

    keys = jax.random.split(sample_key, num_steps)

    @jit
    def _one_step(state: NamedTuple, xs: tuple) -> tuple:
        _, rng_key = xs
        state, info = inference_algorithm.step(rng_key, state)
        return state, (state, info)

    if progress_bar:
        one_step = progress_bar_scan(num_steps)(_one_step)
    else:
        one_step = _one_step

    xs = (jnp.arange(num_steps), keys)
    final_state, (state_history, info_history) = jax.lax.scan(
        one_step, initial_state, xs  # type: ignore
    )
    return final_state, state_history, info_history
