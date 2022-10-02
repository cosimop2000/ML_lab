import numpy as np

# Write a function that takes a 1d numpy array and computes its reverse vector
# (last element becomes the first).
def reverse_vector(rv):
    rv = np.flip(rv)
    return rv


# Given the following square array, compute the product of the elements on its
# diagonal. [[1 3 8] [-1 3 0] [-3 9 2]]
def diag_compute(dg):
    dg = np.diag(dg)
    prodotto = np.prod(dg)
    return prodotto


# Create a random vector of size (3, 6) and find its mean value
def create_random(size1, size2):
    mean1 = 0
    ran = np.random.rand(size1, size2)
    mean1 = np.mean(ran)
    print(ran)
    return mean1


# Given two arrays a and b, compute how many time an item of a is higher than the
# corresponding element of b
def higher(a, b):
    if a.shape != b.shape:
        return 'Different shapes'
    ver = a > b
    cnt = np.count_nonzero(ver)
    print(ver)
    return cnt


# Create and normalize the following matrix (use min-max normalization).
def normalize(m):
    print(m)
    x_min = np.min(m)
    x_max = np.max(m)
    x_scaled = (m - x_min)/(x_max - x_min)
    print(x_scaled)
    return x_scaled


if __name__ == "__main__":
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(a)
    print(a.dtype, a.shape, a.ndim, a.sum(), a.sum(axis=0))
    b = a.reshape((3, 3))
    b = b ** 2
    print(b, b.ndim, b.sum(axis=1))
    c = np.arange(1000)
    d = np.linspace(0, 100, 33)
    print(c, d)
    e = np.ones((5, 4))
    print(e, e+89)
    print(b[0:2, :])
    # slicing dopo la virgola viene fatto sulle colonne, prima viene fatto sulle righe

    # f = b * e[:, np.newaxis]
    # print(f, b.shape, e[:, np.newaxis].shape)

    x = np.random.normal(10, 100, (5, 5))
    print(x)

    y1 = np.random.normal(2, 6, (13,))
    print(y1)
    y1 = reverse_vector(y1)
    print(y1)

    y2 = np.array([[1, 3, 8], [-1, 3, 0], [-3, 9, 2]])
    print(y2, diag_compute(y2))
    #y2 = np.array([[1, 3, 8], [-1, 3, 0], [-3, 9, 2], [1, 1, 1]])
    #print(y2, diag_compute(y2))

    y3 = create_random(3, 6)
    print(y3)

    y4 = np.random.normal(3, 10, (3, 7))
    y5 = np.random.normal(3, 10, (3, 7))
    print(higher(y4, y5))

    y6 = np.array([[0.35, -0.27, 0.56], [0.15, 0.65, 0.42], [0.73, -0.78, -0.08]])
    normalize(y6)
