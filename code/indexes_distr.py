from numpy import sqrt, mean, histogram
import matplotlib.pyplot as plt
import random as rnd
from misc import read_from_file, epsilon_median

# This method is needed for retrieving only the k values from the file
def get_k_values(list_of_vals: list[str]):
    epsilons = []
    k_values = []
    for i in list_of_vals:
        tmp1, tmp2 = i.split('#')
        epsilons.append(float(tmp1))
        k_values.append(float(tmp2))
    
    return epsilons, k_values

''' 
We want to get, for each trial, the index of the element that, given a trend-plot, goes below 1% and 0.1%.
This method is integrated into get_ith_iteration.
'''
def check_index(mu, x_i):
    cond1 = False #1% REVISE THIS LIMITER
    cond2 = False #0.1%
    #NB: the absolute value is important, otherwise, given a normal distributed graph, all the values from one side will be taken
    #Another reminder: we are setting two std limiters, so this is another reason for which abs is important
    if abs( ((x_i/mu) - 1) * 100) < 0.1: 
        cond1 = True
    if abs( ((x_i/mu) - 1) * 100)  < 0.02:
        cond2 = True

    return cond1, cond2


def plot_data_distribution(xvalues, title):
    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    # plots the data histogram: we plot the pre-computed bins and hist by treating each bin
    # as a single point with a weight equal to its count
    ax.hist(x=xvalues, bins=25, color='darkblue', label='data')
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Frequency")
    ax.legend()


def plot_k_barplots(k_values, y_values1, y_values2):    # TO REVISE

    iter_means1 = [mean(sublist) for sublist in y_values1]
    iter_means2 = [mean(sublist) for sublist in y_values2]

    fig, ax = plt.subplots(2, 1)


    ax[0].set_title("iterations which value go under 0.1%")
    ax[0].plot(k_values, iter_means1)
    #ax[0].hist(iter_means1)
    ax[0].set_ylabel("Iteration mean values")

    ax[1].set_title("iterations which value go under 0.02%")
    ax[1].plot(k_values, iter_means2)
    #ax[1].hist(iter_means2)
    ax[1].set_xlabel("K constants")
    ax[1].set_ylabel("Iteration mean values")


# MAIN ENTRY
def main():

    # parameters
    mu = pow(10, 5)
    sigma = sqrt(mu)
    epsilons, k_values = get_k_values(read_from_file('generated_epsilons')) # reads the k constants only from file
    actual_medians = [0.0 for i in range(len(epsilons))] # for each epsilon, we need a separate variable that contains the estimated median


    # index lists
    one_percent = [[] for i in range(len(k_values))]
    one_permille = [[] for i in range(len(k_values))] 

    for stories in range(1000):
        # simulates a stream of 1000 data (1 story)
        for i in range(1000):
            datapoint = rnd.normalvariate(mu, sigma)
                
            for j in range(len(k_values)):
                actual_medians[j] = epsilon_median(datapoint, epsilons[j], i, actual_medians[j])
                cond1, cond2 = check_index(mu, actual_medians[j])
                if cond1 == True: one_percent[j].append(i)
                if cond2 == True: one_permille[j].append(i)
        actual_medians = [0.0 for i in range(len(epsilons))] # actual medians reset at the end of a story
    

    for i in range(len(k_values)):
        plot_data_distribution(one_percent[i], 'Iterations which numbers go below 0.1% for k = {:.0f}, data count = {}'.format(k_values[i], len(one_percent[i])))
        plot_data_distribution(one_permille[i], 'Iterations which numbers go below 0.02% for k = {:.0f}, data count = {}'.format(k_values[i], len(one_permille[i])))

    plot_k_barplots(k_values, one_percent, one_permille)

    plt.show()

if __name__ == "__main__":
    main()