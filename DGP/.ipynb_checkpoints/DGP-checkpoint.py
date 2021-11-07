import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt


def plot_hists(X,y,d):
    fig = matplotlib.pyplot.gcf()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].hist(X[:,0],bins = 50)
    axs[0, 0].set_title('x1')
    axs[0, 1].hist(X[:,1],bins = 50)
    axs[0, 1].set_title('x2')
    axs[0, 2].hist(d)
    axs[0, 2].set_title('Treatment')
    axs[1, 0].hist(X[:,2],bins = 50)
    axs[1, 0].set_title('x3')
    axs[1, 1].hist(X[:,3],bins = 50)
    axs[1, 1].set_title('x4')
    axs[1, 2].hist(y,bins = 50)
    axs[1, 2].set_title('Outcome')

    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

def dgp2(n, var_e):
    np.random.seed()
    p=[1./2, 1./2] #probability of treatment assignment
    d = np.random.choice([0, 1], size=(n,1), p=p) #tr
    x1 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))
    x2 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))
    x3 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))
    x4 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))    
    e = np.random.normal(loc = 0.0, scale = var_e, size = (n,1))
    ind = np.zeros((n,1))
    ind[np.where(x1 >= 0)] = 1
    y = np.add(np.add(np.add(np.add(np.add(d*-1.5, np.multiply(3*d, ind)),x2),x3),x4),e)
    X = np.concatenate((x1,x2,x3,x4), axis=1)
    return X, y, d