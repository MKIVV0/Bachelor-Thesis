from misc import plot_histogram, epsilon_median, read_from_file
from numpy import median as numpy_median, sqrt, mean, std
import TwoHeaps as th
import random as rnd
from matplotlib.pyplot import show as plt_show

# Represents ONE median-estimation trial
# The object parameter is only used when a two-heaps algorithm is passed
# The median_function parameter is used when only a function is needed for estimating the median
# it returns the i-th value for each algorithm run
def get_ith_iteration(mu: float, sigma: float, num_of_iteration: int, median_function1: float, median_function2: float, epsilon_values: list[float]):
    x_i = 0
    tmp_trial_median1: float = 0
    tmp_trial_median2: list[float] = [0.0 for i in range(len(epsilon_values))]
    tmp_trial_median3: float = 0
    tmp_data = []
    tmp_two_heaps = th.TwoHeaps()

    for i in range(1000):
        x_i = rnd.normalvariate(mu, sigma)
        tmp_data.append(x_i)
        tmp_two_heaps.insert(x_i)
        # The epsilon-median is here because it need past values, which
        # are not stored to calculate the current median
        for j in range(len(epsilon_values)):
            tmp_trial_median2[j] = median_function2(
                x_i, epsilon_values[j], i, tmp_trial_median2[j])  # epsilon-median

        if i == num_of_iteration:
            tmp_trial_median1 = median_function1(tmp_data)  # numpy.median
            tmp_trial_median3 = tmp_two_heaps.findMedian()
            break

    return float(tmp_trial_median1), list(tmp_trial_median2), float(tmp_trial_median3)


# The parameters shown in each plots are locally calculated and are not the known ones
def plot_error_histograms(mu, sigma, epsilon_values: list[float], k_values: list[float]):
    num_of_trials = 1000
    ith_values1: list[float] = []  # error values for numpy.median
    ith_values2: list[list[float]] = [[] for i in epsilon_values]  # error values for ε-median1
    ith_values3: list[float] = []  # error values for two-heaps median
    # Executes num_of_trials trials and gets the i-th value for each trial
    # The other values are thrown away
    for i in range(num_of_trials):
        x, y, z = get_ith_iteration(
            mu, sigma, 800, numpy_median, epsilon_median, epsilon_values)
        ith_values1.append(x)
        for index in range(len(ith_values2)):
            ith_values2[index].append(y[index])
        ith_values3.append(z)

    def to_percent_format(x): return ((x/mu) - 1) * 100
    ith_values1 = list(map(to_percent_format, ith_values1))
    for index in range(len(ith_values2)):
        ith_values2[index] = list(map(to_percent_format, ith_values2[index]))
    ith_values3 = list(map(to_percent_format, ith_values3))


    plot_histogram(ith_values1, "Numpy.median error histogram", mean(ith_values1), std(ith_values1), format='percent')
    for index in range(len(ith_values2)): plot_histogram(ith_values2[index], "Epsilon-median percentage deviations histogram, k = {:.0f}, ε = {:.2f}".format(k_values[index], epsilon_values[index]), mean(ith_values2[index]), std(ith_values2[index]), format='percent')
    plot_histogram(ith_values3, "Two-heaps median error histogram", mean(ith_values3), std(ith_values3), format='percent')


# This method is needed for deserilalizing the values from the file generated_epsilons.txt
def split_k_epsilons(list_of_vals: list[str]):
    k_values = []
    epsilons = []
    for i in list_of_vals:
        tmp1, tmp2 = i.split('#')
        k_values.append(float(tmp1))
        epsilons.append(float(tmp2))
    
    return k_values, epsilons
    
# MAIN ENTRY
def main():

    # parameters
    mu = pow(10, 5)
    sigma = sqrt(mu)
    k_values, generated_epsilons = split_k_epsilons(read_from_file())
    plot_error_histograms(mu, sigma, k_values, generated_epsilons)
    plt_show()

if __name__ == "__main__":
    main()