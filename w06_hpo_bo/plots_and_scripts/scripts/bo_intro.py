from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from bo_intro_utils import *
from matplotlib import pyplot as plt
plt.style.use(['ggplot', 'seaborn-talk'])



# Initialize Gaussian Process
kernel = 2.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
gp = GaussianProcessRegressor(kernel=kernel)

# Generate data and fit GP
noise = np.random.rand(4)
data = np.linspace(0.15, 0.9, 4)[:, np.newaxis]
y = np.sin(data[:, 0]*12)+1/2*np.sin(data[:, 0]*11)+1/2*np.sin(data[:, 0]*23)
gp.fit(data, y)

# Plot datapoints
plt.scatter(data[:, 0], y, c='k', marker='X', s=100, zorder=9)
plt.xlim(0, 1)
plt.ylim(-6, 6)
plt.savefig("plot_datapoints.pdf", format='pdf')
plt.show()

X_ = np.linspace(0, 1, 600)
y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)


samples = [1, 3, 10, 100, 1000]
seed = 13

# Plot samples from GP
for i, num_sample in enumerate(samples):
    plot_sample_gp(num_samples=num_sample, data=data, y=y, X_domain=X_, gp=gp, rnd_state=seed)

# Plot GP posterior
plot_posterior_and_density(data=data, y=y, X_=X_, y_mean=y_mean, y_cov=y_cov, gp=gp, rnd_state=seed)



