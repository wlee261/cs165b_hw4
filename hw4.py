#!/usr/bin/env python3
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

def read_data(path):
    """
    Read the input file and store it in data_set.

    DO NOT CHANGE SIGNATURE OF THIS FUNCTION

    Args:
        path: path to the dataset

    Returns:
        data_set: n_samples x n_features
            A list of data points, each data point is itself a list of features.
    """
    data_set = []
    traininglist = []
    training_file = open(path, 'r')
    for x in training_file:
        traininglist.append(x)
    training_file.close()
    for i in range(len(traininglist)):
        data_set.append([])
        line_in_traininglist = traininglist[i].split(",")
        for x in range(len(line_in_traininglist)):
            data_set[i].append(float(line_in_traininglist[x]))
    
    return data_set

def pca(data_set, n_components):
    """
    Perform principle component analysis and dimentinality reduction.
    
    DO NOT CHANGE SIGNATURE OF THIS FUNCTION

    Args:
        data_set: n_samples x n_features
            The dataset, as generated in read_data.
        n_components: int
            The number of components to keep. If n_components is None, all components should be kept.

    Returns:
        components: n_components x n_features
            Principal axes in feature space, representing the directions of maximum variance in the data. 
            They should be sorted by the amount of variance explained by each of the components.
    """

    avg = []
    covar = []
    evs = []

    for i in range(len(data_set[0])):
        count = 0
        total = 0
        for x in range(len(data_set)):
            count = count+1
            total = total + data_set[x][i]
        avg.append(total/count)

    
    for y in range(len(data_set)):
        for z in range(len(data_set[0])):
            data_set[y][z] = data_set[y][z] - avg[z]
    
    covar = np.cov(data_set, rowvar = False)

    evals = []
    evecs = []
    evecs_sorted = []
    evals, evecs = np.linalg.eig(covar)
    evecs_sorted = evecs[:, evals.argsort()]
    for a in range(n_components):
        evs.append(evecs_sorted[:, len(evecs_sorted[:,1]) - a - 1])
    evs = np.transpose(evs)

    return evs
    

    




def dim_reduction(data_set, components):
    """
    perform dimensionality reduction (change of basis) using the components provided.

    DO NOT CHANGE SIGNATURE OF THIS FUNCTION

    Args:
        data_set: n_samples x n_features
            The dataset, as generated in read_data.
        components: n_components x n_features
            Principal axes in feature space, representing the directions of maximum variance in the data. 
            They should be sorted by the amount of variance explained by each of the components.

    Returns:
        transformed: n_samples x n_components
            Return the transformed values.
    """
    return np.dot(data_set, components)


# You may put code here to test your program. They will not be run during grading.
def main():
    data = []
    data = read_data('pizza.txt')
    comp = pca(data, 2)
    result = dim_reduction(data, comp)
    plt.scatter(result[:,0], result[:,1])
    plt.savefig('plot.png')

if __name__ == '__main__':
    main()
