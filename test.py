# %%
import numpy as np
import emcee
import corner
import george
from scipy.stats import shapiro, moment
from statsmodels.tsa.stattools import acf
from scipy.optimize import minimize, dual_annealing
from george.kernels import ConstantKernel, ExpSquaredKernel
from george.modeling import Model, ConstantModel
import matplotlib.pyplot as plt

from corr_coef import r

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
#y_train_noisy = y_train + ar_noise


# %%
class MeanModel(Model):
    parameter_names = ("a", "b")

    def get_value(self, x):
        return self.a * np.power(x, self.b)

    def compute_gradient(self, x):
        return self.a * self.b * np.power(x, self.b - 1)


# %%
process = george.GP(
    ConstantKernel(1, bounds=[(-10, 10)])
    * ExpSquaredKernel(2, metric_bounds=[(-10, 10)]),
    white_noise=ConstantModel(1, bounds=[(-100, 100)]),
    mean=MeanModel(1, 1, bounds=[(-5, 5), (-5, 5)]),
)
process.compute(x_train, yerr=noise_std)


# %%
lam = 1


def lnprob(p):
    process.set_parameter_vector(p)
    _, cov = process.predict(y_train_noisy, x_train, return_cov=True)
    return (
        process.log_likelihood(y_train_noisy, quiet=True)
        + process.log_prior()
        - lam * np.linalg.norm(cov - np.diag(cov))
    )


# %%
result = dual_annealing(
    lambda p: -lnprob(p),
    bounds=process.get_parameter_bounds(),
)
print(result.x)
print(lnprob(result.x))

# %%
plt.plot(x, y, color="k")
plt.errorbar(x_train, y_train_noisy, yerr=noise_std, fmt=".k", capsize=0)

process.set_parameter_vector(result.x)
plt.plot(x, process.predict(y_train, x, return_cov=False))

# %%
plt.figure()
plt.plot(x, MeanModel(*process.get_parameter_vector()[:2]).get_value(x))
plt.plot(x, MeanModel(3, -2).get_value(x))

# %%
res = y_train_noisy - process.predict(y_train, x_train, return_cov=False)

plt.figure()
plt.scatter(x_train, res)
plt.axhline()

# %%
# initial = process.get_parameter_vector()
# ndim, nwalkers = len(initial), 32
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

# p0 = initial + 1e-2 * np.random.randn(nwalkers, ndim)
# sampler.run_mcmc(p0, 20000, progress=True)

# # %%
# tau = sampler.get_autocorr_time()
# flat_samples = sampler.get_chain(discard=10 * int(np.amax(tau)), flat=True)

# # %%
# corner_plot = corner.corner(
#     flat_samples,
#     truths=result.x,
#     quantiles=[0.16, 0.5, 0.84],
#     show_titles=True,
#     labels=process.get_parameter_names(),
# )
# for s in flat_samples[np.random.randint(len(flat_samples), size=24)]:
#     process.set_parameter_vector(s)
#     mu = process.sample_conditional(y_train_noisy, x)
#     plt.plot(x, mu, alpha=0.3)

# %%
