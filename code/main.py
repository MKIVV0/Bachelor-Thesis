import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from scipy.optimize import curve_fit
import TwoHeaps as th
from decimal import *

# ε-median algorithm
def epsilon_median(x_i, epsilon, index, actual_median):
    if index == 0:
        actual_median = x_i
    else:
        sign_num = np.sign(x_i - actual_median)
        actual_median = actual_median + sign_num * epsilon
    return float(actual_median)

'''
We pretend we don't know the distribution parameters and set an arbitrary variable k, which 
is a positive decimal value (e.g. 5%, 10%). k is kept as such in order to investigate how the trend of
the sampling goes for different values of k.
Then, we take the first generated value x_0 and multiply it by said k (k * |x_0|).
In the end, we just plug in the ε-median function.
'''
def get_epsilon(x_0, k):
    return np.abs(x_0) / k


# auxiliary method for renamig the y-axis into a standard deviation format
def update_ylabels(ax, mu, sigma, sigma_coeff):
    ax.set_yticks(np.arange(mu - sigma_coeff*sigma, mu + sigma_coeff*sigma+1, sigma))   # sets the range of the y-axis values
    ylabels = ['{}'.format('μ' if np.ceil(x) == 100000 else str( np.ceil((x-mu)/sigma) ) + 'σ') for x in ax.get_yticks()]  # changes the labels given the condition inside the format
    ax.set_yticklabels(ylabels)


# plots the median estimations for the different algorithms
def plot_median_estimations(x, dset1, dset2, dset3, mu, sigma, epsilon):
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
    ax1.set_ylabel("Estimated medians")
    ax1.set_xlabel("# iterations")
    ax1.scatter(x, dset1)
    update_ylabels(ax1, mu, sigma, sigma_coeff)

    ax2.set_title("ε-median estimations (ε = {})".format(epsilon))
    ax2.grid()
    ax2.axhline(y=100000, color='r')
    ax2.axvline(x=0, color='r')
    ax2.set_ylim(mu - sigma_coeff*sigma, mu + sigma_coeff*sigma)
    ax2.set_ylabel("Estimated medians")
    ax2.set_xlabel("# iterations")
    ax2.scatter(x, dset2)
    update_ylabels(ax2, mu, sigma, sigma_coeff)

    ax3.set_title("Two-heaps median estimations")
    ax3.grid()
    ax3.axhline(y=100000, color='r')
    ax3.axvline(x=0, color='r')
    ax3.set_ylim(mu - sigma_coeff*sigma, mu + sigma_coeff*sigma)
    ax3.set_ylabel("Estimated medians")
    ax3.set_xlabel("# iterations")
    ax3.scatter(x, dset3)
    update_ylabels(ax3, mu, sigma, sigma_coeff)


# plots the histogram of the random generated data
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


# plots the trend of the ε-median algorithm for different epsilons
def plot_epsmedians(x, epsilon_median_lists, epsilonlist, mu, sigma, sigma_coeff):
    num_of_graphs = len(epsilonlist)
    first_half = int(num_of_graphs / 2)
    second_half = num_of_graphs - first_half

    fig1, ax_i1 = plt.subplots(first_half, 1)
    fig1.suptitle("ε-median estimations for different ε - 1st window")

    fig2, ax_i2 = plt.subplots(second_half, 1)
    fig2.suptitle("ε-median estimations for different ε - 2nd window")

    ax_i = np.append(ax_i1, ax_i2)

    for (graph, i) in zip(ax_i, range(len(epsilonlist))):
            if i == (len(epsilonlist) - 1):
                graph.set_title("NEW ε = {:.3f}".format(epsilonlist[i]))  # THIS INDICATES THE ε estimated with the constant k and the value x_0
            else:
                graph.set_title("ε = {}".format(epsilonlist[i]))
                
            graph.grid()
            graph.axhline(y=100000, color='r')
            graph.axvline(x=0, color='r')
            graph.set_ylim(mu - sigma_coeff*sigma, mu + sigma_coeff*sigma)
            graph.set_ylabel("Estimated medians", rotation=90)
            graph.scatter(x, epsilon_median_lists[i])
            update_ylabels(graph, mu, sigma, sigma_coeff)

    # The x label is set only under the last graph of each window, so that it doesn't overlap with the other graphs
    ax_i1[first_half-1].set_xlabel("# iterations")
    ax_i2[second_half-1].set_xlabel("# iterations")


# a = 1 / sqrt(2pi)
# b = x_0, e.g. (x - x_0)^2
# c = 2std^2
# Auxiliary function for finding the gaussian fitting-line on the histogram
def gaussian(x, a, mu, sigma):
    return a*np.exp(-(x - mu)**2/(2*sigma**2))


# MAIN FUNCTION
def main():
    # parameters
    mean = pow(10, 5)
    std = np.sqrt(mean)

    # numpy library median function SECTION
    generated_nums = []  # contains random number generated by the program
    numpy_medians = []  # contains the list of medians

    # ε-median algorithm SECTION
    #epsilon = 0.4
    epsilon1 = [0.01, 0.5, 0.8, 1, 4, 10, 50, 100]
    epsilon_x_i = [0 for i in range(len(epsilon1))]  # this list keeps track of the median evolutions for each constant in epsilon1
    epsilon_medians = [[] for i in range(len(epsilon1))]  # a list of lists, where each sublists has 1000 median-estimations, for the purpose of plotting 

    # GENERATED ε SECTION
    k = 50000   # K CONSTANT FOR THE CALCULATION OF ε
    generated_epsilon = 0
    generated_epsilon_x_i = 0
    generated_epsilon_medians = []

    # two-heaps median algorithm SECTION 
    two_heaps = th.TwoHeaps()
    two_heaps_medians = []

    data_size = pow(10, 3)
    x = np.arange(data_size)  # number of iterations

    # Simulates a random data-flow
    for i in x:
        x_i = rnd.normalvariate(mean, std)
        if i == 0: generated_epsilon = get_epsilon(x_i, k)

        # Standard library median function
        generated_nums.append(x_i)
        numpy_medians.append(np.median(generated_nums))

        # ε median
        #epsilon_x_i = epsilon_median(x_i, epsilon, i, epsilon_x_i)
        #epsilon_medians.append(epsilon_x_i)
        for j in range(len(epsilon1)):
            epsilon_x_i[j] = epsilon_median(x_i, epsilon1[j], i, epsilon_x_i[j])
            epsilon_medians[j].append(epsilon_x_i[j])

        # Generated ε
        generated_epsilon_x_i = epsilon_median(x_i, generated_epsilon, i, generated_epsilon_x_i)
        generated_epsilon_medians.append(generated_epsilon_x_i)

        # Two-heaps median
        two_heaps.insert(x_i)
        two_heaps_medians.append(two_heaps.findMedian())

    # print("Generated nums:", generated_nums)
    # print("Medians:", std_medians)
    # print("EPSILON MEDIAN LISTS:")
    # for i in epsilon_medians:
        #print(i)

    plot_median_estimations(x, numpy_medians, epsilon_medians[1], two_heaps_medians, mean, std, epsilon1[1])
    # plot_histogram(mean, std, generated_nums, 25, 3, data_size)
    # histogram(generated_nums, 25, mean, std)
    plot_histogram(mean, std, generated_nums)
    epsilon_medians.append(generated_epsilon_medians)
    epsilon1.append(generated_epsilon)
    print("TYPES:", type(epsilon_medians), " ", type(epsilon1))
    plot_epsmedians(x, epsilon_medians, epsilon1, mean, std, 2)
    plt.show()


if __name__ == '__main__':
    main()