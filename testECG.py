import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib import pyplot
from scipy import signal

from peakutils.plot import plot as pplot


def read_lable_dict():
    with open('training2017/REFERENCE.csv', mode='r') as infile:
        reader = csv.reader(infile)
        # with open('coors_new.csv', mode='w') as outfile:
        # writer = csv.writer(outfile)
        mydict = {rows[0]: rows[1] for rows in reader}
    return mydict


def plot_line_graph(arrays, labels, title_img):
    # label = label_dict[title_img]
    width = arrays[0].size
    fig = plt.figure(num=None, figsize=(width / 200, 3), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    # ticks = np.arange(0.0, 0.5, 0.02)
    for i in range(len(arrays)):
        array = arrays[i]
        index = range(len(array))
        values = array
        ax.plot(values)  # , colors[i])
        # ax.set_xlabel(tuning_parameter)
        ax.set_ylabel('RMSE Score')
    plt.title(title_img)
    plt.legend(labels, loc='lower right')

    plt.savefig("plots/" + title_img + "_" + label + ".pdf")


for filename in os.listdir('training2017'):
    if filename.endswith('.mat'):
        name = filename[:-4]
        label_dict = read_lable_dict()
        label = label_dict[name]
        mat1 = scipy.io.loadmat('training2017/' + filename)
        # plot_line_graph([mat1['val'][0]], [label], name)
        data = mat1['val'][0]

        # centers = (30.5, 72.3)
        length = len(data)
        x = np.linspace(0, length - 1, length)
        y = data  # (peakutils.gaussian(x, 5, centers[0], 3) +
        # peakutils.gaussian(x, 7, centers[1], 10) +
        # numpy.random.rand(x.size))

        # indexes = peakutils.indexes(y, thres=0.46, min_dist=30)
        # indexes = peakutils.gaussian_fit(x,y, center_only = False)
        # indexes = peakutils.interpolate(x,y)#, func= peakutils.gaussian_fit(x,y))
        # indexes = np.append(indexes, peakutils.indexes(-y, thres=0.45, min_dist=30))
        # print(indexes)
        # print(x[indexes], y[indexes])

        indexes = signal.find_peaks_cwt(y, np.arange(1, 10))
        pyplot.figure(figsize=(10, 6))
        pplot(x, y, indexes)
        pyplot.title('First estimate')

        print "hi"



#
# mat2 = scipy.io.loadmat('A00002.mat')
# mat3 = scipy.io.loadmat('A00003.mat')
# mat4 = scipy.io.loadmat('A00004.mat')
# mat5 = scipy.io.loadmat('A00005.mat')
# mat6 = scipy.io.loadmat('A00006.mat')


print
"done"
