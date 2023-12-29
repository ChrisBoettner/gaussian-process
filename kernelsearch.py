import warnings
from copy import deepcopy
from typing import Optional

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax as ox
from gpjax.base import meta_leaves
from jax import config, jit, tree_map
from jax.stages import Wrapped
from jaxtyping import Array, install_import_hook
from numpy.typing import NDArray
from tqdm import tqdm

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
# Throw error if any of the jax operations evaluate to nan.
config.update("jax_debug_nans", True)

# random seed
key = jr.PRNGKey(42)


class Node:
    def __init__(
        self,
        posterior: gpx.gps.AbstractPosterior,
        max_log_likelihood: Optional[float] = None,
        n_data: Optional[int] = None,
        parent: Optional["Node"] = None,
    ) -> None:
        """Node of the search tree, containg the posterior
        model.

        Parameters
        ----------
        posterior : gpx.gps.AbstractPosterior
            The gpjax posterior model.
        max_log_likelihood : Optional[float], optional
             The log likelihood found after fittng, by default None
        n_data : Optional[int], optional
            The number of datapoints used during training. Needed
            to calculate BIC, by default None
        parent : Optional[Node], optional
            Parent node in the search tree, by default None
        """
        self.children: list["Node"] = []
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
    ) -> None:
        """Update the new by setting new posterior,
        max_log_likelihood and n_posterior.
        Automatically calculates n_parameter from the model,
        and BIC if max_log_likelihood and n_data is available,

        Parameters
        ----------
        posterior : gpx.gps.AbstractPosterior
            The gpjax posterior model.
        max_log_likelihood : Optional[float], optional
             The log likelihood found after fittng, by default None
        n_data : Optional[int], optional
            The number of datapoints used during training. Needed
            to calculate BIC, by default None
        parent : Optional[Node], optional
        """
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
    ) -> None:
        """Add node to list of children.

        Parameters
        ----------
        node : Node
            Node instance of child in search tree.
        """
        self.children.append(node)


class KernelSearch:
    def __init__(
        self,
        kernel_library: list[gpx.kernels.AbstractKernel],
        X: NDArray | Array,
        y: NDArray | Array,
        obs_stddev: float | Array = 1,
        fit_obs_stddev: bool = False,
        likelihood: Optional[gpx.likelihoods.AbstractLikelihood] = None,
        objective: Optional[gpx.objectives.AbstractObjective | Wrapped] = None,
        mean_function: Optional[gpx.mean_functions.AbstractMeanFunction] = None,
        root_kernel: Optional[gpx.kernels.AbstractKernel] = None,
        fitting_mode: str = "scipy",
        num_iters: int = 1000,
        verbosity: int = 1,
    ):
        """Greedy search for optimal Gaussian process kernel structure.

        Parameters
        ----------
        kernel_library : list[gpx.kernels.AbstractKernel]
            List of (initialized) kernel instances that will be used to grow the tree.
        X : NDArray | Array
            Array of X data.
        y : NDArray | Array
            Array of y data.
        obs_stddev : float | Array, optional
            The standard deviation of the y data. Is ignored if custom
            likelihood is given, by default 1
        fit_obs_stddev : bool, optional
            Wether to estimate the standard deviation during fitting process. Is
            ignored if custom likelihood is given, by default False
        likelihood : Optional[gpx.likelihoods.AbstractLikelihood], optional
            Function that calculates the likelihood of the model. By default None,
            which defaults to the Gaussian likelihood with standard deviation
            given by obs_stddev.
        objective : Optional[gpx.objectives.AbstractObjective  |  Wrapped], optional
            The objective function used to evalute the quality of a fit. By default
            None, which defaults to the jit-compiled leave-one-out predictive
            probability.
        mean_function : Optional[gpx.mean_functions.AbstractMeanFunction], optional
            The mean function of the Gaussian process. By default None, which
            sets the mean to zero.
        root_kernel : gpx.kernels.AbstractKernel
            Optional kernel instance that is used as root for the search tree. By
            default None, which uses the kernel library as roots.
        fitting_mode : str, optional
            Fitting procedure. Choose between "scipy" and "adam", by
            default "scipy".
        num_iters : int, optional
            (Maximum) number of iterations for the fitting, by default 1000.
        verbosity : int, optional
            Verbosity of the output between 0 and 2, by default 1
        """

        if isinstance(obs_stddev, float):
            obs_stddev = jnp.array(obs_stddev)

        # set defaults
        if likelihood is None:
            likelihood = gpx.likelihoods.Gaussian(
                num_datapoints=len(X), obs_stddev=obs_stddev
            )
            likelihood = likelihood.replace_trainable(
                obs_stddev=fit_obs_stddev
            )  # type: ignore
        if objective is None:
            objective = jit(gpx.objectives.ConjugateLOOCV(negative=True))
        if mean_function is None:
            mean_function = gpx.mean_functions.Zero()

        self.likelihood = likelihood
        self.objective = objective
        self.data = gpx.Dataset(X=X, y=y)
        self.kernel_library = kernel_library

        self.fitting_mode = fitting_mode
        self.num_iters = num_iters

        self.verbosity = verbosity

        # create root node
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
    def _const_kernel(trainable: bool = False) -> gpx.kernels.Constant:
        """Create constant kernel. Used for tracking the different terms
        in the composite kernel.

        Parameters
        ----------
        trainable : bool, optional
            Choose wether the constant should be learned during
            fitting, by default False

        Returns
        -------
        gpx.kernels.Constant
            Constant kernel.
        """
        return gpx.kernels.Constant(constant=jnp.array(1.0)).replace_trainable(
            constant=trainable
        )  # type: ignore

    def fit(
        self, posterior: gpx.gps.AbstractPosterior
    ) -> tuple[gpx.gps.AbstractPosterior, float]:
        """Fit the hyperparameter of a posterior. Can be done using
        scipy's 'minimize' function using the 'adam' optimiser from
        optax.

        Parameters
        ----------
        posterior : gpx.gps.AbstractPosterior
            Posterior model object containing the hyperparameter.

        Returns
        -------
        tuple[gpx.gps.AbstractPosterior, float]
            Returns the posterior with optimised hyperparameter and
            log likelihood found at maximum.

        Raises
        ------
        ValueError
            Thrown if optimiser mode is unknown.
        """
        if self.fitting_mode == "scipy":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                optimized_posterior, history = gpx.fit_scipy(
                    model=posterior,
                    objective=self.objective,
                    train_data=self.data,
                    max_iters=self.num_iters,
                    verbose=self.verbosity >= 2,
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
                verbose=self.verbosity >= 2,
            )
        else:
            raise ValueError("'fitting_mode' must be 'scipy' or 'adam'.")

        max_log_likelihood = -float(history[-1])
        return optimized_posterior, max_log_likelihood

    def expand_node(self, node: Node) -> None:
        """Logic to expand a node in the search tree. The
        child nodes are created by adding or multiplying
        the kernels from the kernel library to the current
        kernel. New kernels are added to children of
        the Node instance.

        Parameters
        ----------
        node : Node
            Current Node instance to expand.
        """
        for kernel_operation in [gpx.kernels.ProductKernel, gpx.kernels.SumKernel]:
            for ker in self.kernel_library:
                kernel = deepcopy(node.posterior.prior.kernel)

                new_kernel = deepcopy(ker)  # type: ignore
                if kernel_operation == gpx.kernels.SumKernel:
                    # create new additive term with tracer constant,
                    # the first kernel in the new term has a trainable constant
                    new_kernel = gpx.kernels.ProductKernel(
                        kernels=[self._const_kernel(), new_kernel]
                    )  # type: ignore
                if kernel_operation == gpx.kernels.ProductKernel:
                    # further kernels have variance fixed, so that we only
                    # have one multiplicative constant per term
                    try:
                        new_kernel = new_kernel.replace_trainable(
                            variance=False
                        )  # type: ignore
                    except ValueError:
                        pass

                composite_kernel = kernel_operation(
                    kernels=[kernel, new_kernel]
                )  # type: ignore

                new_prior = gpx.gps.Prior(
                    mean_function=node.posterior.prior.mean_function,
                    kernel=composite_kernel,
                )
                new_posterior = self.likelihood * new_prior
                node.add_child(Node(new_posterior, parent=node))

    def compute_layer(
        self,
        layer: list[Node],
        current_depth: int,
    ) -> None:
        """Fit the hyperparameter of the posterior for
        all nodes in the current layer.

        Parameters
        ----------
        layer : list[Node]
            List of nodes in the current layer.
        current_depth : int
            Integer depth of current layer, used for
            tracking.
        """
        for node in tqdm(
            layer,
            desc=f"Fitting Layer {current_depth +1}",
            disable=False if self.verbosity == 1 else True,
        ):
            node.update(*self.fit(node.posterior), self.data.n)

    def select_top_nodes(
        self,
        layer: list[Node],
        bic_threshold: float,
        n_leafs: int,
    ) -> list[Node]:
        """Select top nodes of current layer, based on
        their BIC value.

        Parameters
        ----------
        layer : list[Node]
            List of nodes in the current layer.
        bic_threshold : float
            Maximum BIC nodes that to hit to be considered
            for next layer. Usually minimum (best) BIC of
            previous layer.
        n_leafs : int
            Number top nodes to keep.

        Returns
        -------
        list[Node]
            Sorted list of top nodes.
        """
        sorted_tuple = sorted((node.bic, node) for node in layer)
        # return first n_leafs nodes
        top_nodes = [node for _, node in sorted_tuple][:n_leafs]
        # filter for bic threshold
        top_nodes = [node for node in top_nodes if node.bic < bic_threshold]
        if top_nodes:
            self.best_model = top_nodes[0]
        return top_nodes

    def expand_layer(
        self,
        layer: list[Node],
    ) -> list[Node]:
        """Expand nodes in the current layer.

        Parameters
        ----------
        layer : list[Node]
            Layer of current (top) nodes.

        Returns
        -------
        list[Node]
            List of nodes in next layer.
        """
        next_layer = []
        for node in layer:
            self.expand_node(node)
            next_layer.extend(node.children)
        return next_layer

    def search(
        self,
        depth: int = 10,
        n_leafs: int = 3,
    ) -> gpx.gps.AbstractPosterior:
        """Search for the best kernel fitting the data
        by performing a greedy search through possible kernel
        combinations.
        Start with simple kernel, which gets progressively more
        complex by adding or multiplying new kernels from kernel
        library. Kernels are evaluated by calculating their BIC
        after being fit to data.

        Parameters
        ----------
        depth : int, optional
            The number of layers of the search tree. Deeper layers
            correspond to more complex kernels, by default 10
        n_leafs : int, optional
            The number of kernels to keep and expand at each layer. Top
            kernels are chosen based on BIC, by default 3

        Returns
        -------
        gpx.gps.AbstractPosterior
            The (fitted) gpjax posterior object for the best kernel.
        """
        layer = self.root

        current_depth = 0
        bic_threshold = np.inf
        for current_depth in range(depth):
            # fit and compute BIC at current layer
            self.compute_layer(layer, current_depth)
            if current_depth == 0:
                best_model = sorted((node.bic, node) for node in layer)[0][1]

            # calculate and sort BICs
            current_bics = sorted([node.bic for node in layer])
            if self.verbosity >= 1:
                print(f"Layer {current_depth+1} || Current BICs: {current_bics}")

            # Early stopping if no more improvements are found
            if current_bics[0] > bic_threshold:
                if self.verbosity >= 1:
                    print("No more improvements found! Terminating early..\n")
                    break

            # expand tree and search for new top noded in next layer down
            layer = self.select_top_nodes(layer, bic_threshold, n_leafs)
            bic_threshold = current_bics[0]  # min bic of current layer
            best_model = layer[0]
            layer = self.expand_layer(layer)

        if self.verbosity >= 1:
            print(f"Terminated on layer: {current_depth+1}.")
            print(f"Final log likelihood: {best_model.max_log_likelihood}")
            print(f"Final number of model paramter: {best_model.n_parameter}")
        return best_model.posterior
