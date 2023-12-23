# %%
import numpy as np


def r(A):
    # Ensure A is a numpy array
    A = np.array(A)

    # Check if A is a square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square (d x d).")
    if (A < 0).any():
        raise ValueError("Matrix must have no negative entries.")

    d = A.shape[0]

    # Precomputed vectors
    j = np.ones(d)
    r = np.arange(1, d + 1)
    r2 = r**2

    # Precompute matrix-vector products that are used more than once
    Aj = np.dot(A, j)
    Ar = np.dot(A, r)

    # Compute various sums more efficiently
    n = np.sum(A)
    sum_x = np.dot(r, Aj)
    sum_y = np.dot(j, Ar)
    sum_x_2 = np.dot(r2, Aj)
    sum_y_2 = np.dot(np.dot(j, A), r2)
    sum_xy = np.dot(r, Ar)

    # Calculating the square root expressions more efficiently
    sigma_x = np.sqrt(n * sum_x_2 - sum_x**2)
    sigma_y = np.sqrt(n * sum_y_2 - sum_y**2)

    r = (n * sum_xy - sum_x * sum_y) / (sigma_x * sigma_y)

    return r


# %%
A = np.diag([1, 2, 31, 0.4, 5])
B = np.random.randint(10, size=[10, 10])
print(r(A))
# %%
