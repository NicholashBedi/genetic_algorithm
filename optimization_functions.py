import numpy as np

def get_prob_cross_selection(number_selected):
    probs = np.zeros(number_selected)
    sum_of_numbers_selected = number_selected*(number_selected+1)/2
    for i in range(number_selected):
        probs[i] = (number_selected - i)/sum_of_numbers_selected

def limit_range(x, x_range):
    return max(min(x, x_range[1]), x_range[0])
