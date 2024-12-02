import numpy as np
import matplotlib.pyplot as plt
from parser import get_args
args = get_args()

def plot_histogram(x_list, title, range_tuple=None, label_list=None):
    assert len(np.array(x_list).shape)==2
    range_tuple = range_tuple if range_tuple else (
        np.quantile(x_list, 0.01, axis=1).min(),
        np.quantile(x_list, 0.99, axis=1).max()
    )
    ymax = plt.hist(x_list, bins=20, range=range_tuple, label=label_list, stacked=False)[0].max()
    if len(x_list)==1:
        plt.vlines(np.quantile(x_list[0], 0.25), 0, ymax, colors='orange', linestyle='dashed')
        plt.vlines(np.quantile(x_list[0], 0.50), 0, ymax, colors='orange', linestyle='solid' )
        plt.vlines(np.quantile(x_list[0], 0.75), 0, ymax, colors='orange', linestyle='dashed')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(f'{args.output_path}/histogram_{title}.png')
    plt.clf()
    plt.close()