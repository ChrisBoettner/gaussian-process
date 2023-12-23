# %%
import numpy as np
import emcee
import corner
import george
from scipy.stats import shapiro, moment
from statsmodels.tsa.stattools import acf
from scipy.optimize import minimize, dual_annealing
from george.kernels import (
    ConstantKernel,
    ExpSquaredKernel,
    ExpKernel,
    Matern32Kernel,
    DotProductKernel,
    LinearKernel,
)
from george.modeling import Model, ConstantModel
import matplotlib.pyplot as plt

rng = np.random.RandomState(2)

# %%

x = np.linspace(1, 10, 1000)
y = 3 * np.power(x, -2) + 2 * np.sin(1 / 3 * x)

training_indices = rng.choice(np.arange(y.size), size=60, replace=False)
x_train, y_train = x[training_indices], y[training_indices]

# %%
noise_std = 0.5
noise = rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)

rho = 0.9
ar_noise = [0]
for i in range(len(noise) - 1):
    ar_noise.append(rho * ar_noise[i] + noise[i])
ar_noise_std = np.sqrt(noise_std**2 / (1 - rho**2))

y_train_noisy = y_train + noise
# y_train_noisy = y_train + ar_noise

# %%
kernel_lib = [
    ExpSquaredKernel,
    ExpKernel,
    Matern32Kernel,
    DotProductKernel,
]

extended_kernel_lib = [ConstantKernel(1)] + [
    ConstantKernel(1) * kernel(1) for kernel in kernel_lib
]
for i in range(len(kernel_lib)):
    for j in range(i + 1, len(kernel_lib)):
        extended_kernel_lib.append(
            ConstantKernel(1) * kernel_lib[i](1) * kernel_lib[j](1)
        )

full_kernel = None
for kernel in extended_kernel_lib:
    if full_kernel is None:
        full_kernel = kernel
    else:
        full_kernel += kernel


# %%
process = george.GP(
    full_kernel,
    white_noise=ConstantModel(1),
)
process.compute(x_train, yerr=noise_std)


# %%
lam = 10


def lnprob(p):
    process.set_parameter_vector(p)
    constant_names = ["log_constant" in name for name in process.get_parameter_names()]
    constants = process.get_parameter_vector()[constant_names]
    return (
        process.log_likelihood(y_train_noisy, quiet=True)
        + process.log_prior()
        - lam * np.sum(np.abs(constants))
    )


# %%
result = minimize(
    lambda p: -lnprob(p),
    x0=[1 for l in range(len(process.get_parameter_vector()))],
    bounds=process.get_parameter_bounds(),
)
print(result.x)
print(lnprob(result.x))

# %%
parameter_vector = np.copy(result.x)
process.set_parameter_vector(result.x)

keep = 4
constants = np.sort(
    np.abs(
        parameter_vector[
            [("log_constant" in name) for name in process.get_parameter_names()]
        ]
    )
)[::-1]
threshold = constants[keep - 1]

neglectable_terms = np.array(
    ["log_constant" in name for name in process.get_parameter_names()]
) * np.array([np.abs(val) < threshold for val in process.get_parameter_vector()])

parameter_vector[neglectable_terms] = -np.inf
print(parameter_vector)


# %%

process.set_parameter_vector(parameter_vector)
plt.plot(x, y, color="k")
plt.errorbar(x_train, y_train_noisy, yerr=noise_std, fmt=".k", capsize=0)

plt.plot(x, process.predict(y_train, x, return_cov=False))
# %%
