import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from scipy.optimize import curve_fit
import TwoHeaps as th
import time

'''
We pretend we don't know the distribution parameters and set an arbitrary variable k, which
is a positive decimal value (e.g. 5%, 10%). k is kept as such in order to investigate how the trend of
the sampling goes for different values of k.
Then, we take the first generated value x_0 and multiply it by said k (k * |x_0|).
In the end, we just plug in the ε-median function.
'''
# Auxiliary function
def get_epsilon(x_0, k): return float(np.abs(x_0) / k)


# a = 1 / sqrt(2pi)
# b = x_0, e.g. (x - x_0)^2
# c = 2std^2
# Auxiliary function for finding the gaussian fitting-line on the histogram
def gaussian(x, a, mu, sigma):
    return a*np.exp(-(x - mu)**2/(2*sigma**2))


# auxiliary method for renamig the y-axis into a standard deviation format
def update_ylabels(ax, mu, sigma, sigma_coeff):
    # sets the range of the y-axis values
    ax.set_yticks(np.arange(mu - sigma_coeff*sigma,
                  mu + sigma_coeff*sigma+1, sigma))
    ylabels = ['{:.3f}%'.format( np.abs(((x/mu) - 1) * 100) )
               for x in ax.get_yticks()]  # changes the labels given the condition inside the format
    ax.set_yticklabels(ylabels)


# ε-median algorithm
def epsilon_median(x_i, epsilon, index, actual_median):
    if index == 0:
        actual_median = x_i
    else:
        sign_num = np.sign(x_i - actual_median)
        actual_median = actual_median + sign_num * epsilon
    return float(actual_median)


# plots the median estimations for the different algorithms
def plot_median_estimations(x, dset1, dset2, dset3, mu, sigma, epsilon):
    sigma_coeff = 3
   # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle(
        'Median estimations over iterations\nμ: {}\nσ: {:.2f}'.format(mu, sigma))
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
def plot_histogram(data, title):
    local_mean = np.mean(data)
    local_std = np.std(data)
    # values of the histogram, bin edges (len(hist)+1)
    hist, bin_edges = np.histogram(data, 25)
    # calculates the bin centers,by taking pairs, summing them and multiply them by .5
    bin_centers = [.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(bin_edges)-1)]
    
    # fitting curve function
    popt, pcov = curve_fit(f=gaussian, xdata=bin_centers, ydata=hist, p0=[max(hist), local_mean, local_std])
    print(popt, '\n', pcov)

    fig, ax = plt.subplots(1, 1)
    ax.set_title("{}:\nμ: {:.2f}\nσ: {:.2f}".format(
        title, local_mean, local_std))
    # plots the fitting line curve
    ax.plot(bin_centers, gaussian(bin_centers, *popt), 'r', label='gaussian fit')
    # plots the data histogram: we plot the pre-computed bins and hist by treating each bin
    # as a single point with a weight equal to its count
    ax.hist(x=bin_edges[:-1], bins=bin_edges, weights=hist, color='darkblue', label='data')
    ax.set_xlabel("Data")
    ax.set_ylabel("Frequency")
    ax.legend()

    # Converts probabilities back to frequencies
    #ylabels = ['{}'.format(int(x * len(data))) for x in ax.get_yticks()]
    #ax.set_yticklabels(ylabels)


# plots the trend of the ε-median algorithm for different epsilons
def plot_epsmedians(x, epsilon_median_lists, epsilonlist, k, mu, sigma, sigma_coeff):
    num_of_graphs = len(epsilonlist)
    first_half = int(num_of_graphs / 2)
    second_half = num_of_graphs - first_half

    fig1, ax_i1 = plt.subplots(first_half, 1)
    fig1.suptitle("ε-median estimations for different ε - 1st window")

    fig2, ax_i2 = plt.subplots(second_half, 1)
    fig2.suptitle("ε-median estimations for different ε - 2nd window")

    ax_i = np.append(ax_i1, ax_i2)

    for (graph, i) in zip(ax_i, range(len(epsilonlist))):
        graph.set_title("k = {:.2e}   ε = {:.2f}".format(k[i], epsilonlist[i]))

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


# Represents ONE median-estimation trial
# The object parameter is only used when a two-heaps algorithm is passed
# The median_function parameter is used when only a function is needed for estimating the median
# it returns the i-th value for each algorithm run

def get_ith_iteration(mean, std, num_of_iteration, median_function1, median_function2, epsilon):
    x_i = 0
    tmp_trial_median1 = 0
    tmp_trial_median2 = 0
    tmp_trial_median3 = 0
    tmp_data = []
    tmp_two_heaps = th.TwoHeaps()
    for i in range(1000):
        x_i = rnd.normalvariate(mean, std)
        tmp_data.append(x_i)
        tmp_two_heaps.insert(x_i)
        # The epsilon-median is here because it need past values, which
        # are not stored to calculate the current median
        tmp_trial_median2 = median_function2(
            x_i, epsilon, i, tmp_trial_median2)  # epsilon-median
        if i == num_of_iteration:
            tmp_trial_median1 = median_function1(tmp_data)  # numpy.median
            tmp_trial_median3 = tmp_two_heaps.findMedian()
            break

    return float(tmp_trial_median1), float(tmp_trial_median2), float(tmp_trial_median3)


def plot_error_histograms(mean, std, epsilon):
    num_of_trials = 1000
    ith_values1 = []  # error values for numpy.median
    ith_values2 = []  # error values for ε-median1
    ith_values3 = []  # error values for two-heaps median
    # Executes num_of_trials trials and gets the i-th value for each trial
    # The other values are thrown away
    for i in range(num_of_trials):
        x, y, z = get_ith_iteration(
            mean, std, 800, np.median, epsilon_median, epsilon)
        ith_values1.append(x)
        ith_values2.append(y)
        ith_values3.append(z)

    print("VALUES\n")
    print(ith_values2[0:10])
    print(ith_values3[0:10])

    def to_percent_format(x): return (x/mean) - 1
    ith_values1 = list(map(to_percent_format, ith_values1))
    ith_values2 = list(map(to_percent_format, ith_values2))
    ith_values3 = list(map(to_percent_format, ith_values3))

    fig, ax_i = plt.subplots(1, 3)
    fig.tight_layout(pad=2.5)
    fig.suptitle("Percentage deviation with ε = {:.3f}".format(epsilon))

    for ax in ax_i:
        ax.title.set_text("Numpy.median error histogram")
        ax.set_title("mean: {:e}\nstd: {:e}".format(
            np.mean(ith_values1), np.std(ith_values1)))
        ax.hist(ith_values1, color='darkblue')

        ax.title.set_text("Epsilon-median error histogram")
        ax.set_title("mean: {:e}\nstd: {:e}".format(
            np.mean(ith_values2), np.std(ith_values2)))
        ax.hist(ith_values2, color='darkblue')

        ax.title.set_text("Two-heaps median error histogram")
        ax.set_title("mean: {:e}\nstd: {:e}".format(
            np.mean(ith_values3), np.std(ith_values3)))
        ax.hist(ith_values3, color='darkblue')



# MAIN FUNCTION
# This main executes: the three algorithm confrontation, histogram plotting with gaussian-fitting,
# epsilon-median trends for different epsilons
def main():
    # parameters
    mean = pow(10, 5)
    std = np.sqrt(mean)

    # numpy library median function SECTION
    generated_nums = []  # contains random number generated by the program
    numpy_medians = []  # contains the list of medians

    # ε-median algorithm SECTION
    # epsilon = 0.4
    #epsilon1 = [0.01, 0.5, 0.8, 1, 4, 10, 50, 100]
    # this list keeps track of the median evolutions for each constant in epsilon1
    #epsilon_x_i = [0 for i in range(len(epsilon1))]
    # a list of lists, where each sublists has 1000 median-estimations, for the purpose of plotting
    #epsilon_medians = [[] for i in range(len(epsilon1))]

    # GENERATED ε SECTION
    #k = 200000   # K CONSTANT FOR THE CALCULATION OF ε
    k = [250, 500, 750, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5]  # the arbitrary constants that serve for the calulation of ε
    generated_epsilons = []  # stores the epsilon calculated with the different constants of the array above
    generated_epsilon_x_i = [0 for i in range(len(k))] # this array serves for calculating the epsilon-medians, which are cumulative on the same variable
    generated_epsilon_medians = [[] for i in range(len(k))]  # stores the trends for the various epsilons

    # two-heaps median algorithm SECTION
    two_heaps = th.TwoHeaps()
    two_heaps_medians = []

    data_size = pow(10, 3)
    x = np.arange(data_size)  # number of iterations

    th_tmp_var = 0  # TWO HEAPS TMP VAR
    delta_t_two_heaps = 0

    # Simulates a random data-flow
    for i in x:
        x_i = rnd.normalvariate(mean, std)
        if i == 0:
            for constant in k:  # the epsilons for different ks are generated
                generated_epsilons.append(get_epsilon(x_i, constant))  

        # Standard library median function
        generated_nums.append(x_i)
        numpy_medians.append(np.median(generated_nums))

        # ε median
        # epsilon_x_i = epsilon_median(x_i, epsilon, i, epsilon_x_i)
        # epsilon_medians.append(epsilon_x_i)
        #for j in range(len(epsilon1)):
        #    epsilon_x_i[j] = epsilon_median(
        #        x_i, epsilon1[j], i, epsilon_x_i[j])
        #   epsilon_medians[j].append(epsilon_x_i[j])

        # Generated ε
        '''
        for epsilon, index in zip(generated_epsilons, range(len(generated_epsilons)) ):
            actual_epsilon_median = epsilon_median(x_i, epsilon, i, actual_epsilon_median)
            generated_epsilon_x_i[index] = actual_epsilon_median
            generated_epsilon_medians[index].append(actual_epsilon_median)
        '''
        for j in range(len(generated_epsilons)):
            generated_epsilon_x_i[j] = epsilon_median(x_i, generated_epsilons[j], i, generated_epsilon_x_i[j])
            generated_epsilon_medians[j].append(generated_epsilon_x_i[j])    

        '''
        Here we calculate the execution time for retrieving the median
        at the 10th iteration, including all the insertion needed beforehand.
        The explaination continues below, on top of the line that prints 
        "NUMPY VS TWO-HEAPS:"
        '''
        # Two-heaps median
        # t1
        t1 = time.time()
        two_heaps.insert(x_i)
        # t2
        t2 = time.time()
        delta_t_two_heaps = delta_t_two_heaps + (t2 - t1)
        if i == 9:
            th_tmp_var = two_heaps.findMedian()
        two_heaps_medians.append(two_heaps.findMedian())
        
        

    # This section represents the error histogram plot
    for eps in generated_epsilons:
        plot_error_histograms(mean, std, eps)
    plot_median_estimations(x, numpy_medians, generated_epsilon_medians[7], two_heaps_medians, mean, std, generated_epsilons[7])
    # plot_histogram(mean, std, generated_nums, 25, 3, data_size)
    # histogram(generated_nums, 25, mean, std)
    plot_histogram(generated_nums, "Generated data distribution")
    #epsilon_medians.append(generated_epsilon_medians)
    #epsilon1.append(generated_epsilon)
    plot_epsmedians(x, generated_epsilon_medians, generated_epsilons, k, mean, std, 2)
    plt.show()

    ''' 
    Test for veryfying the time it takes for both the numpy.median and 
    the two heaps algorithm to retrieve the median.
    DISCLAIMER: We suppose that the numpy.median function orders the array and 
    takes the median value. 
    '''
    print("NUMPY VS TWO-HEAPS:\n")
    t1 = time.time()
    print(np.sort(generated_nums[:10]))
    t2 = time.time()
    delta_t_numpy = t2 - t1
    print(th_tmp_var)
    print(numpy_medians[9])

    print("\nExecution times:\nNumpy: {:e}\nTwo-heaps: {:e}".format(
        delta_t_numpy, delta_t_two_heaps))


if __name__ == '__main__':
    main()
