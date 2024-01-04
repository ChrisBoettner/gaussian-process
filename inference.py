import gpjax as gpx
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from beartype.typing import Callable, NamedTuple
from blackjax.base import SamplingAlgorithm
from blackjax.progress_bar import progress_bar_scan
from blackjax.types import ArrayLikeTree
from gpjax.typing import Array, ScalarFloat
from jax import jit
from gpjax.objectives import ConjugateMLL
from jax.tree_util import tree_leaves

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

    - Calculate marginal probability for GP outside of transit
    (background, using mask)
    - Remove model lightcurve
    - Calculate probability of light curve data after
    removing the transit (inverse mask), with the GP conditioned
    on the background data

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

    D_background = gpx.Dataset(
        X=X[mask].reshape(-1, 1),
        y=y[mask].reshape(-1, 1),
    )
    D_transit = gpx.Dataset(
        X=X[~mask].reshape(-1, 1),
        y=y[~mask].reshape(-1, 1),
    )

    marginal_log_likelihood = ConjugateMLL()

    # fix gp variables to to initial values
    if fix_gp:
        # constrain the posterior
        updated_posterior = gp_posterior.constrain()

        # marginal log likelihood for background,
        # constitutes likelihood function for hyperparameter
        # conditioned on data outside of transit
        background_log_prob = marginal_log_likelihood(
            updated_posterior,
            D_background,
        )

        transit_dist = calculate_predictive_dist(
            updated_posterior,
            D_transit.X,  # type: ignore
            D_background,
        )

        def objective(params: Array) -> ScalarFloat:
            # calculate lightcurve model
            lightcurve = lc_model(D_transit.X, params["lc_parameter"])

            # remove lightcurve from observations and
            # calculate probability under GP model
            res = (D_transit.y - lightcurve).reshape(-1)
            transit_log_prob = transit_dist.log_prob(res.reshape(-1))
            # return (negative of, if wanted) log probabilitys
            return constant * jnp.nan_to_num(
                transit_log_prob + background_log_prob, nan=-jnp.inf
            )

    # adapt gp parameter at every step
    else:
        # indices of trainables for GP
        trainable_idx = jnp.argwhere(
            jnp.array(tree_leaves(gp_posterior.trainables()))
        ).reshape(-1)

        def objective(params: Array) -> ScalarFloat:
            # update the parameter of the posterior object
            updated_posterior = jit_set_trainables(
                gp_posterior,
                jnp.array(params["gp_parameter"]),
                trainable_idx,
            ).constrain()

            # marginal log likelihood for background,
            # constitutes likelihood function for hyperparameter
            # conditioned on data outside of transit
            background_log_prob = marginal_log_likelihood(
                updated_posterior,
                D_background,
            )

            transit_dist = calculate_predictive_dist(
                updated_posterior,
                D_transit.X,  # type: ignore
                D_background,
            )
            # calculate lightcurve model
            lightcurve = lc_model(D_transit.X, params["lc_parameter"])

            # remove lightcurve from observations and
            # calculate probability under GP model
            res = D_transit.y - lightcurve
            transit_log_prob = transit_dist.log_prob(res.reshape(-1))

            # return (negative of, if wanted) log probability
            return constant * jnp.nan_to_num(
                transit_log_prob + background_log_prob, nan=-jnp.inf
            )

    if compile:
        return jit(objective)
    return objective


@jit
def calculate_predictive_dist(
    posterior: gpx.gps.AbstractPosterior,
    input: Array,
    train_data: gpx.Dataset,
) -> tfp.distributions.Distribution:
    """Calculate the predictive distribution
    of the GP for a given input (x data), under
    the observations given by train_data.

    Parameters
    ----------
    posterior : gpx.gps.AbstractPosterior
        The GPJax posterior object.
    input : Array
        The (x) values at which to calculate the
        distribution
    train_data : gpx.Dataset
        The training data to condition the GP on.

    Returns
    -------
    tfp.distributions.Distribution
        The predictive distribution. (Most likely
        multivariate Gaussian)
    """
    latent_dist = posterior(
        input,
        train_data=train_data,
    )
    predictive_dist = posterior.likelihood(
        latent_dist
    )  # adds observational uncertainty
    return predictive_dist


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
