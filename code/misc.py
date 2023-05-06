from numpy import arange, abs, mean, std, exp, histogram, sign, sqrt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from os import sep, path, remove


# ε-median algorithm
def epsilon_median(x_i, epsilon, index, actual_median):
    if index == 0:
        actual_median = x_i
    else:
        sign_num = sign(x_i - actual_median)
        actual_median = actual_median + sign_num * epsilon
    return float(actual_median)


# auxiliary method for renamig the y-axis into a standard deviation format
def update_ylabels(ax, mu, sigma, sigma_coeff):
    # sets the range of the y-axis values
    ax.set_yticks(arange(mu - sigma_coeff*sigma,
                  mu + sigma_coeff*sigma+1, sigma))
    ylabels = ['{:.3f}%'.format( abs(((x/mu) - 1) * 100) )
               for x in ax.get_yticks()]  # changes the labels given the condition inside the format
    ax.set_yticklabels(ylabels)


def write_to_file(list_of_values: list[float], filename='generated_epsilons'):
    path_locator = '.' + sep
    folder_name = 'data_folder' + sep
    full_path = path_locator + folder_name + filename

    if path.exists(full_path):
        remove(full_path)
    
    with open(full_path, 'w') as f:
        for value in list_of_values:
            f.write(str(value) + '\n')
    
    f.close()


def read_from_file(filename='generated_epsilons'):
    path_locator = '.' + sep
    folder_name = 'data_folder' + sep

    list_of_values = []

    with open(path_locator + folder_name + filename, 'r') as reader:
        for row in reader:
            list_of_values.append(float(row))   

        reader.close()

    return list_of_values


# a = 1 / sqrt(2pi)
# b = x_0, e.g. (x - x_0)^2
# c = 2std^2
# Auxiliary function for finding the gaussian fitting-line on the histogram
def gaussian(x, a, mu, sigma):
    return a*exp(-(x - mu)**2/(2*sigma**2))


# plots the histogram of the random generated data
def plot_histogram(data, title):
    local_mean = mean(data)
    local_std = std(data)
    # values of the histogram, bin edges (len(hist)+1)
    hist, bin_edges = histogram(data, 25)
    # calculates the bin centers,by taking pairs, summing them and multiply them by .5
    bin_centers = [.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(bin_edges)-1)]
    
    # fitting curve function
    popt, pcov = curve_fit(f=gaussian, xdata=bin_centers, ydata=hist, p0=[max(hist), local_mean, local_std])
    #print(popt, '\n', pcov)
 
    fig, ax = plt.subplots(1, 1)
    ax.set_title("{}:\nμ: {:.5e}\nσ: {:.5e}".format(
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

# FOR DEBUGGING PURPOSES
def main():
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    write_to_file(a)

    numbers = read_from_file()
    print(numbers)

if __name__ == "__main__":
    main()