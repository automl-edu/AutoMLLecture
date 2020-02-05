import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use(['ggplot', 'seaborn-talk'])


# Get list containing minimums of GP samples
def get_mins(samples=None):
    num_samples = samples.shape[1]
    if num_samples > 1:
        mins = []
        for sample in range(num_samples):
            mins.append(np.argmin(samples[:, sample]))
    else:
        mins = np.argmin(samples)
    return mins

# Plot sample from posterior and histogram over minimum
def plot_sample_gp(num_samples=10, data=None, y=None, X_domain=None, gp=None, rnd_state=0):
    # Plot sample from posterior

    label = ["Sample Curve"]
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
    y_samples = gp.sample_y(X_domain[:, np.newaxis], num_samples, random_state=rnd_state)
    plots = a0.plot(X_domain, y_samples, lw=1, label=label)
    a0.scatter(data[:, 0], y, c='k', marker='X', s=100, zorder=9)
    a0.legend([plots[0]], label)
    a0.set_ylabel("f(x)")
    a0.set_xlabel("x")
    a0.set_xlim(0, 1)
    a0.set_ylim(-6, 6)


    mins = get_mins(y_samples)

    sns.distplot(X_domain[mins, np.newaxis], hist=True, kde=False, bins=50, norm_hist=True, hist_kws=dict(edgecolor='k', color='#6BAFFC'))
    a1.set_xlim(0, 1)
    a1.set_ylabel("Pmin(x)")
    a1.set_xlabel("x")
    plt.yticks([])
    f.tight_layout()
    plt.savefig("plot_posterior_%s_sample.pdf" % num_samples, format='pdf')
    plt.show()


# Plot GP posterior and density over minimum
def plot_posterior_and_density(data=None, y=None,  X_=None, y_mean=None, y_cov=None, gp=None, rnd_state=0):
    # Plot GP posterior
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
    uncertainty = 3.5 * np.sqrt(np.diag(y_cov))
    a0.plot(X_, y_mean, color='#0F028A', linewidth=2, alpha=0.8, label="GP mean")
    a0.fill_between(X_, y_mean - uncertainty, y_mean + uncertainty, alpha=0.3, facecolor='lightblue', edgecolor='k', label="GP variance")
    a0.scatter(data[:, 0], y, c='k', marker='X', s=100, zorder=9)
    a0.set_xlim(0, 1)
    a0.set_ylim(-6, 6)
    a0.set_ylabel("f(x)")
    a0.set_xlabel("x")
    a0.legend()

    # Plot density over minimums from samples
    y_samples = gp.sample_y(X_[:, np.newaxis], 1000, rnd_state)
    mins = get_mins(y_samples)

    sns.distplot(X_[mins, np.newaxis], kde=True, hist=False, bins=50, color='#6BAFFC')
    a1.set_ylabel('Pmin(x)')
    a1.set_xlabel("x")

    a1.set_xlim(0, 1)
    plt.yticks([])
    f.tight_layout()
    plt.savefig("plot_posterior.pdf", format='pdf')
    plt.show()
