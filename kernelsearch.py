# Enable Float64 for more stable matrix inversions.
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

from jax import jit, tree_map
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook, Array
from copy import deepcopy
from typing import Optional
import numpy as np
from gpjax.base import meta_leaves
from jax.stages import Wrapped
import warnings
import optax as ox
from tqdm import tqdm

import matplotlib.pyplot as plt

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    from gpjax.kernels import Constant
    
key = jr.PRNGKey(42)

class Node:
    def __init__(
        self,
        posterior: gpx.gps.AbstractPosterior,
        max_log_likelihood: Optional[float] = None,
        n_data: Optional[int] = None,
        parent=Optional["Node"],
    ):
        self.children = []
        self.parent = parent

        self.update(
            posterior,
            max_log_likelihood,
            n_data,
        )

    def update(
        self,
        posterior: gpx.gps.AbstractPosterior,
        max_log_likelihood: Optional[float] = None,
        n_data: Optional[int] = None,
    ):
        self.posterior = posterior

        self.n_parameter = sum(
            leaf[0]["trainable"]
            for leaf in meta_leaves(posterior)
            if isinstance(leaf[0], dict)
        )  # number of trainable parameter

        if n_data is not None:
            self.n_data = np.log(n_data)

        if max_log_likelihood is not None:
            self.max_log_likelihood = max_log_likelihood
            if self.n_data is not None:
                self.bic = self.n_parameter * self.n_data - 2 * self.max_log_likelihood

    def add_child(
        self,
        node: "Node",
    ):
        self.children.append(node)


class KernelSearch:
    def __init__(
        self,
        kernel_library: list[gpx.kernels.AbstractKernel],
        data: gpx.Dataset,
        obs_stddev: float | Array = 1,
        fit_obs_stddev: bool = False,
        likelihood: Optional[gpx.likelihoods.AbstractLikelihood] = None,
        objective: Optional[gpx.objectives.AbstractObjective | Wrapped] = None,
        mean_function: Optional[gpx.mean_functions.AbstractMeanFunction] = None,
        root_kernel: Optional[gpx.kernels.AbstractKernel] = None,
        fitting_mode: str = "scipy",
        num_iters: int = 1000,
        parallelise: bool = True,
        n_cores: int = 8,
        verbosity: int = 1,
    ):
        """_summary_

        Parameters
        ----------
        kernel_library : list[gpx.kernels.AbstractKernel]
            _description_
        data : gpx.Dataset
            _description_
        obs_stddev : float | Array, optional
            _description_, is ignored if custom likelihood is given, by default 1
        fit_obs_stddev : bool, optional
            _description_, is ignored if custom likelihood is given, by default False
        likelihood : Optional[gpx.likelihoods.AbstractLikelihood], optional
            _description_, by default None, which defaults to the Gaussian likelihood with standard deviation given by obs_stddev.
        objective : Optional[gpx.objectives.AbstractObjective  |  Wrapped], optional
            _description_, by default None, which defaults to the jit-compiled leave-one-out predictive probability 
        mean_function : Optional[gpx.mean_functions.AbstractMeanFunction], optional
            _description_, by default None, which sets the mean to zero
        root_kernel : gpx.kernels.AbstractKernel
        _description_
        fitting_mode : str, optional
            _description_, by default "scipy"
        num_iters : int, optional
            _description_, by default 1000
        parallelise : bool, optional
            _description_, by default True
        n_cores : int, optional
            _description_, by default 8
        verbosity : int, optional
            _description_, by default 1
        """
        if isinstance(obs_stddev, float):
            obs_stddev = jnp.array(obs_stddev)
        if likelihood is None:
            likelihood = gpx.likelihoods.Gaussian(
                num_datapoints=data.n, obs_stddev=obs_stddev
            )
            likelihood = likelihood.replace_trainable(obs_stddev=fit_obs_stddev)  # type: ignore
        if objective is None:
            objective = jit(gpx.objectives.ConjugateLOOCV(negative=True))
        if mean_function is None:
            mean_function = gpx.mean_functions.Zero()

        self.likelihood = likelihood
        self.objective = objective
        self.data = data
        self.kernel_library = kernel_library

        self.fitting_mode = fitting_mode
        self.num_iters = num_iters

        self.parallelise = parallelise
        self.n_cores = n_cores

        self.verbosity = verbosity

        self.root = [
            Node(
                likelihood
                * gpx.gps.Prior(
                    mean_function=mean_function,
                    kernel=self._const_kernel() * kernel,
                )
            )
            for kernel in (kernel_library if root_kernel is None else [root_kernel])
        ]


    @staticmethod
    def _const_kernel(trainable=False):
        return Constant(constant=jnp.array(1.0)).replace_trainable(constant=trainable)  # type: ignore

    def fit(self, posterior) -> tuple[gpx.gps.AbstractPosterior, float]:
        if self.fitting_mode == "scipy":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                optimized_posterior, history = gpx.fit_scipy(
                    model=posterior,
                    objective=self.objective,
                    train_data=self.data,
                    max_iters=self.num_iters,
                    verbose=self.verbosity >= 2
                )
        elif self.fitting_mode == "adam":
            static_tree = tree_map(lambda x: not x, posterior.trainables)
            optim = ox.chain(
                ox.adamw(learning_rate=3e-4),
                ox.masked(
                    ox.set_to_zero(),
                    static_tree,
                ),
            )
            optimized_posterior, history = gpx.fit(
                model=posterior,
                objective=self.objective,
                train_data=self.data,
                optim=optim,
                key=key,
                num_iters=self.num_iters,
                verbose=self.verbosity >= 2
            )
        else:
            raise ValueError("'fitting_mode' must be 'scipy' or 'adam'.")

        max_log_likelihood = -float(history[-1])
        return optimized_posterior, max_log_likelihood

    def expand_node(self, node):
        for kernel_operation in [gpx.kernels.ProductKernel, gpx.kernels.SumKernel]:
            for ker in self.kernel_library:
                kernel = deepcopy(node.posterior.prior.kernel)

                new_kernel = deepcopy(ker)  # type: ignore
                if kernel_operation == gpx.kernels.SumKernel:
                    # create new additive term with tracer constant
                    # the first kernel in the new term has a trainable constant
                    new_kernel = gpx.kernels.ProductKernel(kernels=[self._const_kernel(), new_kernel])  # type: ignore
                if kernel_operation == gpx.kernels.ProductKernel:
                    # further kernels have variance fixed, so that we only have one constant
                    try:
                        new_kernel = new_kernel.replace_trainable(variance=False)  # type: ignore
                    except ValueError:
                        pass

                composite_kernel = kernel_operation(kernels=[kernel, new_kernel])  # type: ignore

                new_prior = gpx.gps.Prior(
                    mean_function=node.posterior.prior.mean_function,
                    kernel=composite_kernel,
                )
                new_posterior = self.likelihood * new_prior
                node.add_child(Node(new_posterior, parent=node))

    def compute_layer(self, layer, current_depth):
        if self.verbosity == 1:
            for node in tqdm(layer, desc=f"Fitting Layer {current_depth +1}"):
                node.update(*self.fit(node.posterior), self.data.n)
        else:
            [node.update(*self.fit(node.posterior), self.data.n) for node in layer]

    def select_top_nodes(self, layer, bic_threshold, n_leafs):
        sorted_tuple = sorted((node.bic, node) for node in layer)
        # return first n_leafs nodes
        top_nodes = [node for _, node in sorted_tuple][:n_leafs]
        # filter for bic threshold
        top_nodes = [node for node in top_nodes if node.bic < bic_threshold]
        if top_nodes:
            self.best_model = top_nodes[0]
        return top_nodes

    def expand_layer(self, layer):
        next_layer = []
        for node in layer:
            self.expand_node(node)
            next_layer.extend(node.children)
        return next_layer

    def search(
        self,
        depth: int = 10,
        n_leafs: int = 3,
    ):
        layer = self.root

        current_depth = 0
        bic_threshold = np.inf
        for current_depth in range(depth):
            self.compute_layer(layer, current_depth)
            if current_depth == 0:
                best_model = sorted((node.bic, node) for node in layer)[0][1]

            current_bics = sorted([node.bic for node in layer])
            if self.verbosity >= 1:
                print(f"Layer {current_depth+1} || Current BICs: {current_bics}")

            if current_bics[0] > bic_threshold:
                if self.verbosity >= 1:
                    print("No more improvements found! Terminating early..\n")
                    break

            layer = self.select_top_nodes(layer, bic_threshold, n_leafs)
            bic_threshold = current_bics[0]  # min bic of current layer
            best_model = layer[0]
            layer = self.expand_layer(layer)

        if self.verbosity >= 1:
            print(f"Terminated on layer: {current_depth+1}.")
            print(f"Final log likelihood: {best_model.max_log_likelihood}")
            print(f"Final number of model paramter: {best_model.n_parameter}")
        return best_model.posterior