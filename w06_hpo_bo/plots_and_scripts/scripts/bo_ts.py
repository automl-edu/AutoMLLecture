from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import norm
import numpy as np

plt.style.use(['ggplot', 'seaborn-talk'])

seed = 41
# for seed in range(0, 100):
#     print("Seed is ", seed)
np.random.seed(seed)


# Toy objective function
def poly(x):
    return -(3 * np.sin(20*x) - 5 * np.cos(30*x) - 10*np.power(x, 2))


# Initialize Gaussian Process
kernel = 2.0 * RBF(length_scale=5e-1, length_scale_bounds=(1e-1, 1e2))
gp = GaussianProcessRegressor(kernel=kernel)

xbounds = [-1.0, 1.0]
ybounds = [-5, 8]
# Generate data and fit GP
data = np.array([-0.45, -0.18, 0.26])
data = data[:, np.newaxis]
y = poly(data[:, 0])
gp.fit(data, y)


def plot_objf():
    X_ = np.linspace(xbounds[0], xbounds[1], 600)
    ys = poly(X_)
    plt.plot(X_, ys, color='#0ABFBF', linewidth=1.5, alpha=0.4)


def plot_posterior(data=None, y=None,  X_=None, y_mean=None, y_cov=None, kappa=3.5, lcb_only=False):
    uncertainty = kappa * np.sqrt(np.diag(y_cov))
    ax.plot(X_, y_mean, color='#0F028A', linewidth=2, alpha=0.8)
    if lcb_only:
        ax.fill_between(X_, y_mean - uncertainty, y_mean, alpha=0.3, facecolor='lightblue', edgecolor='k')
    else:
        ax.fill_between(X_, y_mean - uncertainty, y_mean + uncertainty, alpha=0.3, facecolor='lightblue', edgecolor='k')
    ax.scatter(data[:, 0], y, c='k', marker='X', s=100, zorder=9)
    # ax.set_xlim(0, 1)
    # ax.set_ylim(-6, 6)

# Plot basic GP
def plot_gp_base(lcb_only=False):
    X_ = np.linspace(xbounds[0], xbounds[1], 600)
    y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
    plot_posterior(data=data, y=y, X_=X_, y_mean=y_mean, y_cov=y_cov, kappa=0.5, lcb_only=lcb_only)
    plt.yticks([])
    plt.xticks([])
    plt.xlim(xbounds)
    plt.ylim(ybounds)


def highlight_xplus(xy):
    recto = (xbounds[0], xy[1])
    rectwidth = xbounds[1] - xbounds[0]
    rectheight = ybounds[1] - xy[1]
    rect = Rectangle(recto, rectwidth, rectheight, fill=True, alpha=0.5, color='black')
    plt.gca().add_patch(rect)


def get_bell_curve_xy(mu=0.0, sigma=1.0, step=0.01, xlims=(-10.0, 10.0), yscale=1.0):
    xs = np.arange(xlims[0], xlims[1], step)
    ys = norm.pdf(xs, mu, sigma) * yscale
    # print(xs.size, ys.size)
    return xs, ys


def draw_vertical_normal(xtest=0.0, label="x", mu=0.0, sigma=1.0, step=0.01, xlims=(10.0, 10.0), yscale=1.0):
    # Generate a normal pdf centered at xtest
    ytest_mean, ytest_cov = gp.predict([[xtest]], return_cov=True)
    ytest_mean = ytest_mean[0]
    ytest_cov = ytest_cov[0, 0]
    # print("ytest mean:{}, cov:{}".format(ytest_mean, ytest_cov))
    norm_x, norm_y = get_bell_curve_xy(mu=mu, sigma=sigma, step=step, xlims=xlims, yscale=yscale)

    # Rotate by -pi/2 to obtain a vertical curve
    vcurve_x = norm_y + xtest
    vcurve_y = -norm_x + ytest_mean

    # plt.plot(xtest, ytest_mean, c='grey', marker='X', zorder=10)
    # plt.vlines(xtest, ymin=ybounds[0], ymax=ybounds[1], colors='k', linestyles='dashed', linewidths=0.5, zorder=10)
    plt.plot(vcurve_x, vcurve_y, c='k', linewidth=0.5, zorder=10)
    fill_args = np.where(vcurve_y < fxplus)
    plt.fill_betweenx(vcurve_y[fill_args], xtest, vcurve_x[fill_args], alpha=0.8, facecolor='darkgreen', zorder=10)
    # plt.annotate(s='$({0}, \mu({0}))$'.format(label), xy=(xtest, ytest_mean), xytext=(xtest - 0.2, ytest_mean - 1.0),
    #              arrowprops={'arrowstyle': 'fancy'}, weight='heavy', fontsize='x-large', zorder=10)
    # plt.annotate(s='$PI({})$'.format(label), xy=(xtest + 0.1, ytest_mean), xytext=(xtest + 0.3, ytest_mean - 1.0),
    #              arrowprops={'arrowstyle': 'fancy'}, weight='heavy', fontsize='x-large', color='darkgreen', zorder=10)


def draw_labelled_vline(x, label):
    plt.vlines(x, ymin=ybounds[0], ymax=ybounds[1], colors='k', linestyles='dashed', linewidths=0.5, zorder=10)
    plt.annotate(s=label, xy=(x, ybounds[0]), xytext=(x, ybounds[0]), weight='heavy', fontsize='x-large')


def draw_labelled_hline(y, label):
    plt.hlines(y, xmin=xbounds[0], xmax=xbounds[1], colors='k', linestyles='dashed', linewidths=0.5, zorder=10)
    plt.annotate(s=label, xy=(xbounds[0], y), xytext=(xbounds[0], y), weight='heavy', fontsize='x-large')


def scale_gp_sample(sample_ys, mean_ys, scale=0.5):
    # If gp samples are too wild, tame them to something closer to the mean
    diff = mean_ys - sample_ys
    return mean_ys - scale * diff


def draw_gp_sample(npoints=500, scale=0.5, color='#CF020A', seed=0):
    X_ = np.linspace(xbounds[0], xbounds[1], npoints)[:, np.newaxis]
    y = gp.sample_y(X_, random_state=seed)
    if scale not in [1.0, 0.0]:
        if scale > 1.0:
            raise RuntimeWarning("Are you sure you want to scale beyond 1.0?")
        mean_ys = gp.predict(X_, return_std=False, return_cov=False)[:, np.newaxis]
        y_draw = scale_gp_sample(y, mean_ys, scale)
    elif scale == 1.0:
        print("Drawing original gp sample")
        y_draw = y
    else:
        print("Re-drawing mean")
        y_draw = gp.predict(X_, return_std=False, return_cov=False)[:, np.newaxis]

    plt.plot(X_, y_draw[:, 0], color=color, linewidth=2, alpha=0.6)


f, ax = plt.subplots(1, 1, squeeze=True)
f.tight_layout()
plot_gp_base()
plt.savefig("ts_1.pdf", format="pdf")
sample_seed = 100
draw_gp_sample(scale=0.3, seed=sample_seed)
#plt.show()
plt.savefig("ts_2.pdf", format="pdf")

xmin = 0.689
fxmin = -1.8
draw_labelled_vline(x=xmin, label="$x_t$")
draw_labelled_hline(y=fxmin, label="$g(x_t)$")
plt.plot(xmin, fxmin, c='g', marker='X', zorder=10)
plt.savefig("ts_3.pdf", format="pdf")
plt.show()
