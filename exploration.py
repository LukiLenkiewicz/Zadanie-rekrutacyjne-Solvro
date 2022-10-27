import numpy as np
import matplotlib.pyplot as plt
from constants import CLASSES, NUM_OF_CLASSES


def calculate_distance(arr):
    distances = []
    for i in range(len(arr)-1):
        distances.append(np.linalg.norm(arr[i]-arr[i+1]))
    
    return distances


def plot_samples(samples_per_class, title, figure_size=(20, 15), plot_type="plot"):
    num_of_samples_per_class = len(samples_per_class[0])
    fig, axis = plt.subplots(NUM_OF_CLASSES, num_of_samples_per_class, figsize=figure_size)
    fig.suptitle(title)

    for i, class_ in enumerate(samples_per_class):
        for j, sample in enumerate(class_):
            if plot_type == "plot":
                axis[i, j].plot(sample[:, 0], sample[:, 1])
            elif plot_type == "plot_distances":
                distance = calculate_distance(sample)
                axis[i, j].plot(distance)
            elif plot_type == "histogram":
                distance = calculate_distance(sample)
                axis[i, j].hist(distance)
            axis[i, j].set_title(f"class: {CLASSES[i]}, sample no.: {j+1}")

    plt.show()