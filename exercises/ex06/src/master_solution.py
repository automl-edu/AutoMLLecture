import numpy as np
import matplotlib.pyplot as plt
import operator
import time

# ################################################### Data to use ######################################################
train_data = np.array([(np.array(x), np.sin(x) + 0.1 * np.cos(10 * x)) for x in np.linspace(0, 2 * np.pi, 64)])
X = np.array([np.array(x) for x in np.linspace(0, 2 * np.pi, 16, endpoint=False)])
x, y = (zip(*train_data))
# ################################################### Data to use ######################################################


def phi_n_of_x(n: int) -> np.ndarray:
    """
    Basis function factory. Allows for easy generation of \phi_n(x).
    :param n: Dimensionality of the basis function
    :return: \phi_n(x)
    """

    def phi_of_x(x):
        return np.array([x ** i for i in range(n)])

    return phi_of_x


def gp_prediction_a(data: np.ndarray, X: np.ndarray, phi: callable, sigma_n: float, Sigma_p: np.ndarray) \
        -> (np.ndarray, np.ndarray):
    """
    Implementation of 2.11
    :param data: Data on which the model is fit
    :param X: Data to predict the mean and variance
    :param phi: basis functions
    :param sigma_n: variance for points in data
    :param Sigma_p: covariance
    :return: mean and variance for all points in X
    """
    # precompute data specific matrices
    Phi_T = np.vstack(list(map(phi, map(operator.itemgetter(0), data))))
    A = np.dot(np.transpose(Phi_T), Phi_T) / (sigma_n ** 2) + np.linalg.inv(Sigma_p)
    A_inv = np.linalg.inv(A)
    y = np.array(list(map(operator.itemgetter(1), data)))
    # handy short hand notation
    mm = A_inv.dot(np.transpose(Phi_T).dot(y)) / (sigma_n ** 2)

    # for storing the results
    means = []
    variances = []

    # loop over all X
    for xs in X:
        phi_x = phi(xs)
        means.append(np.dot(phi_x, mm))
        variances.append(np.dot(phi_x, np.dot(A_inv, phi_x)))

    return np.array(means), np.array(variances)


def gp_prediction_b(data: np.ndarray, X: np.ndarray, phi: callable, sigma_n: float, Sigma_p: np.ndarray) \
        -> (np.ndarray, np.ndarray):
    """
    Implementation of 2.12
    :param data: Data on which the model is fit
    :param X: Data to predict the mean and variance
    :param phi: basis functions
    :param sigma_n: variance for points in data
    :param Sigma_p: covariance
    :return: mean and variance for all points in X
    """
    # precompute data specific matrices
    y = np.array(list(map(operator.itemgetter(1), data)))
    Phi_T = np.vstack(list(map(phi, map(operator.itemgetter(0), data))))
    K = np.dot(Phi_T, np.dot(Sigma_p, np.transpose(Phi_T)))
    inv = np.linalg.inv(K + sigma_n ** 2 * np.eye(K.shape[0]))

    means = []
    variances = []

    for xs in X:
        phi_x = phi(xs)
        means.append(phi_x.dot(Sigma_p.dot((Phi_T.T).dot(inv.dot(y)))))
        variances.append(phi_x.dot(Sigma_p.dot(phi_x)) - phi_x.dot(
            Sigma_p.dot((Phi_T.T).dot(inv.dot(Phi_T.dot(Sigma_p.dot(phi_x)))))))

    return np.array(means), np.array(variances)


# ##################################################### Exercise (b)
# Plot model predictions for the bayesian linear regression:
num_features = 2
means, variances = gp_prediction_a(train_data, x, phi_n_of_x(num_features), 1.0, np.eye(num_features))

plt.fill_between(x, means - 2 * np.sqrt(variances), means + 2 * np.sqrt(variances), alpha=0.2)
plt.plot(x, means, label='posterior mean')
plt.scatter(x, y, label='observations')
plt.title('Bayesian linear regression example')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
# plt.savefig('plot1.pdf')


# ##################################################### Exercise (c)
# compare the predictions
mean1, var1 = gp_prediction_a(train_data, X, phi_n_of_x(num_features), 1.0, np.eye(num_features))
mean2, var2 = gp_prediction_b(train_data, X, phi_n_of_x(num_features), 1.0, np.eye(num_features))
print('Equal means: ', np.allclose(mean1, mean2))
print('Equal variances: ', np.allclose(var1, var2))
print()

# plot the computation time
tsa = []
tsb = []
dimensions = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
for n in dimensions:
    print('Computing in feature dimension: ', n)
    I_n = np.eye(n)        # setup Identity of dimensionality n
    phi_n = phi_n_of_x(n)  # get basis functions
    start = time.time()
    means, variances = gp_prediction_a(train_data, X, phi_n, 1.0, I_n)
    run_time = time.time() - start
    tsa.append(run_time)
    start = time.time()
    gp_prediction_b(train_data, X, phi_n, 1.0, I_n)
    run_time = time.time() - start
    tsb.append(run_time)

plt.figure()
plt.loglog(dimensions, tsa, label='Implementation 2.11', drawstyle='steps-post')
plt.loglog(dimensions, tsb, label='Implementation 2.12', drawstyle='steps-post')
plt.xlabel('input dimension')
plt.ylabel('runtime[sec]')
plt.legend(loc=2)

# plt.savefig('plot2.pdf')
plt.show()
