from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import norm
import numpy as np

plt.style.use(['ggplot', 'seaborn-talk'])

seed = 41
# for seed in range(0, 100):
#     print("Seed is ", seed)
np.random.seed(seed)


# Toy Objective Function
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


def plot_posterior(ax=None, data=None, y=None,  X_=None, y_mean=None, y_cov=None, kappa=3.5):
    """
    :param ax: matplotlib.axes.Axes object to be used for plotting. If None, a new fig is created and returned.
    :param data:
    :param y:
    :param X_:
    :param y_mean:
    :param y_cov:
    :param kappa: Uncertainty envelope size in number of standard deviations.
    :return: If ax was None, a matplotlib.figure.Figure object, else None
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, squeeze=True)
    else:
        f = None

    uncertainty = kappa * np.sqrt(np.diag(y_cov))
    ax.plot(X_, y_mean, color='#0F028A', linewidth=2, alpha=0.8)
    ax.fill_between(X_, y_mean - uncertainty, y_mean + uncertainty, alpha=0.3, facecolor='lightblue', edgecolor='k')
    ax.scatter(data[:, 0], y, c='k', marker='X', s=100, zorder=9)
    # ax.set_xlim(0, 1)
    # ax.set_ylim(-6, 6)
    return f


def plot_objf():
    X_ = np.linspace(xbounds[0], xbounds[1], 600)
    ys = poly(X_)
    plt.plot(X_, ys, color='#0ABFBF', linewidth=1.5, alpha=0.4)


# Plot basic GP
def plot_gp_base():
    X_ = np.linspace(xbounds[0], xbounds[1], 600)
    y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
    f = plot_posterior(data=data, y=y, X_=X_, y_mean=y_mean, y_cov=y_cov, kappa=0.5)
    plt.yticks([])
    plt.xticks([])
    plt.xlim(xbounds)
    plt.ylim(ybounds)
    f.tight_layout()


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


plot_gp_base()
# plt.savefig("ei_1.pdf", format="pdf")

xplus = -0.18
fxplus = poly(xplus)
draw_labelled_vline(x=xplus, label="$x^+_t$")
draw_labelled_hline(y=fxplus, label="$f(x^+_t)$")
plt.plot(xplus, fxplus, c='b', marker='X', zorder=10)
# plt.annotate(s="$(x^+_t, f(x^+_t))$", xy=(xplus, fxplus), xytext=(xplus-0.1, fxplus-2.0), arrowprops={'arrowstyle': 'fancy'},
#              weight='heavy', fontsize='x-large')
plt.savefig("ei_2.pdf", format="pdf")

highlight_xplus((xplus, fxplus))
plt.savefig("ei_3.pdf", format="pdf")

x1 = 0.7
fx1 = poly(x1)
draw_labelled_vline(x=x1, label="$x$")
draw_labelled_hline(y=fx1, label="$f(x)$")
plt.plot(x1, fx1, c='r', marker='X', zorder=10)
plt.savefig("ei_4.pdf", format="pdf")

xmid = (xplus + x1)/2
plt.annotate(s='', xy=(xmid, fx1), xytext=(xmid, fxplus), arrowprops={'arrowstyle': '<|-|>', 'lw': 3, 'color': 'black'})
plt.text(xmid + 0.02, (fxplus + fx1) / 2, '$I(x)$', weight='heavy', fontsize='x-large')
plt.savefig("ei_5.pdf", format="pdf")

mux1 = gp.predict(np.array(x1).reshape(1, -1))[0]
draw_labelled_hline(y=mux1, label='$\mu(x)$')
plt.plot(x1, mux1, c='g', marker='X', zorder=10)
draw_vertical_normal(xtest=x1, label="x", mu=0.0, sigma=1.0, step=0.01, xlims=[-10.0, 10.0], yscale=0.5)
plt.savefig("ei_6.pdf", format="pdf")

plt.annotate(s='Use for $\mathbb{E}[I(x)]$', xy=(x1 + 0.05, mux1-0.8), xytext=(x1 - 0.3, mux1 - 2.0),
             arrowprops={'arrowstyle': '->', 'lw': 1, 'color': 'black'}, weight='heavy', fontsize='x-large', color='darkgreen', zorder=10)
plt.savefig("ei_7.pdf", format="pdf")

# plt.annotate(s='', xy=(1.5 * xmid, mux1), xytext=(1.5 * xmid, fxplus), arrowprops={'arrowstyle': '<|-|>', 'lw': 3, 'color': 'black'})
# plt.text(1.5 * xmid + 0.02, (fxplus + mux1) / 2, '$\mathbb{E}[I(x_1)]$', weight='heavy', fontsize='x-large')
# plt.savefig("ei_8.pdf", format="pdf")

plt.show()
