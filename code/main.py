import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from scipy.optimize import curve_fit
import TwoHeaps as th


def epsilon_median(x_i, epsilon, index, actual_median):
    if index == 0:
        actual_median = x_i
    else:
        sign_num = np.sign(x_i - actual_median)
        actual_median = actual_median + sign_num * epsilon
    return actual_median

def update_ylabels(ax, mu, sigma, sigma_coeff):
    ax.set_yticks(np.arange(mu - sigma_coeff*sigma, mu + sigma_coeff*sigma+1, sigma))   # sets the range of the y-axis values
    ylabels = ['{}'.format('μ' if np.ceil(x) == 100000 else str( np.ceil((x-mu)/sigma) ) + 'σ') for x in ax.get_yticks()]  # changes the labels given the condition inside the format
    ax.set_yticklabels(ylabels)

def plot_median_estimations(x, dset1, dset2, dset3, mu, sigma):
    sigma_coeff = 3
   # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle('Median estimations over iterations\nμ: {}\nσ: {:.2f}'.format(mu, sigma))
    fig.tight_layout(pad=3.0)

    ax1.set_title("Numpy library median estimations")
    ax1.grid()
    ax1.axhline(y=100000, color='r')
    ax1.axvline(x=0, color='r')
    ax1.set_ylim(mu - sigma_coeff*sigma, mu + sigma_coeff*sigma)
    ax1.set_ylabel("Generated data")
    ax1.set_xlabel("# iterations")
    ax1.scatter(x, dset1)
    update_ylabels(ax1, mu, sigma, sigma_coeff)

    ax2.set_title("Epsilon median estimations")
    ax2.grid()
    ax2.axhline(y=100000, color='r')
    ax2.axvline(x=0, color='r')
    ax2.set_ylim(mu - sigma_coeff*sigma, mu + sigma_coeff*sigma)
    ax2.set_ylabel("Generated data")
    ax2.set_xlabel("# iterations")
    ax2.scatter(x, dset2)
    update_ylabels(ax2, mu, sigma, sigma_coeff)

    ax3.set_title("Two-heaps median estimations")
    ax3.grid()
    ax3.axhline(y=100000, color='r')
    ax3.axvline(x=0, color='r')
    ax3.set_ylim(mu - sigma_coeff*sigma, mu + sigma_coeff*sigma)
    ax3.set_ylabel("Generated data")
    ax3.set_xlabel("# iterations")
    ax3.scatter(x, dset3)
    update_ylabels(ax3, mu, sigma, sigma_coeff)


def plot_histogram(mean, std, x_data):
    # Computes the histogram
    # outputs: values of the histogram, bin edges
    hist, bin_edges = np.histogram(x_data)
    hist = hist/sum(hist)
    n = len(hist)

    # Extracts the x-axis (histogram bin) and y-axis (values)
    # returns an array filled with zeros of length n
    x_hist = np.zeros((n), dtype=float)
    for i in range(n):
        x_hist[i] = (bin_edges[i+1] + bin_edges[i])/2
    y_hist = hist

    # Least-square fitting process on x_hist and y_hist
    param_optimised, param_covariance_matrix = curve_fit(
        gaussian, x_hist, y_hist, p0=[max(y_hist), mean, std])

    # Plots the Gaussian curve
    fig, ax = plt.subplots()
    x_hist_2 = np.linspace(np.min(x_hist), np.max(x_hist), 500)
    ax.plot(x_hist_2, gaussian(x_hist_2, *param_optimised),
             'r-', label='Gaussian fit')  # Plots the fitting line
    ax.legend()

    # Normalize the histogram values
    weights = np.ones_like(x_data) / len(x_data)
    # Plots the data
    ax.hist(x_data, weights=weights)
    ax.set_title("Generated data distribution:\nμ: {:.2f}\nσ: {:.2f}".format(np.mean(x_data), np.std(x_data)))
    ax.set_xlabel("Data")
    ax.set_ylabel("Frequency")

    # Converts probabilities back to frequencies
    ylabels = ['{}'.format( int(x * len(x_data)) ) for x in ax.get_yticks()]
    ax.set_yticklabels(ylabels)


# a = 1 / sqrt(2pi)
# b = x_0, e.g. (x - x_0)^2
# c = 2std^2
def gaussian(x, a, mu, sigma):
    return a*np.exp(-(x - mu)**2/(2*sigma**2))


def main():
    # parameters
    mean = pow(10, 5)
    std = np.sqrt(mean)

    # For standard library median
    generated_nums = []  # contains random number generated by the program
    numpy_medians = []  # contains the list of medians

    # For epsilon median functions
    epsilon = 0.4
    epsilon_x_i = 0
    epsilon_medians = []

    # For two heaps median
    two_heaps = th.TwoHeaps()
    two_heaps_medians = []

    data_size = pow(10, 3)
    x = np.arange(data_size)  # number of iterations

    for i in x:
        x_i = rnd.normalvariate(mean, std)

        # Standard library median function
        generated_nums.append(x_i)
        numpy_medians.append(np.median(generated_nums))

        # Epsilon median
        epsilon_x_i = epsilon_median(x_i, epsilon, i, epsilon_x_i)
        epsilon_medians.append(epsilon_x_i)

        # Two-heaps median
        two_heaps.insert(x_i)
        two_heaps_medians.append(two_heaps.findMedian())

    # print("Generated nums:", generated_nums)
    # print("Medians:", std_medians)

    plot_median_estimations(x, numpy_medians, epsilon_medians, two_heaps_medians, mean, std)
    # plot_histogram(mean, std, generated_nums, 25, 3, data_size)
    # histogram(generated_nums, 25, mean, std)
    plot_histogram(mean, std, generated_nums)
    plt.show()


if __name__ == '__main__':
    main()
