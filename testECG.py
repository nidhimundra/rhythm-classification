import copy
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib import pyplot

import peakutils
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


def func(x, y):
    return peakutils.gaussian_fit(x, y, center_only=False)

for filename in os.listdir('training2017'):
    if filename.endswith('.mat'):
        name = filename[:-4]
        label_dict = read_lable_dict()
        label = label_dict[name]
        mat1 = scipy.io.loadmat('training2017/' + filename)
        # plot_line_graph([mat1['val'][0]], [label], name)
        data = mat1['val'][0]

        # centers = (30.5, 72.3)
        # length = len(data)
        # x = np.linspace(0, length - 1, length)
        y = data  # (peakutils.gaussian(x, 5, centers[0], 3) +
        # peakutils.gaussian(x, 7, centers[1], 10) +
        # numpy.random.rand(x.size))


        minus = 0
        plus = 0
        for i in y:
            if i > 300:
                plus += 1
            if i < -300:
                minus += 1
        if minus > plus:
            y = -y

        yold = copy.copy(y)

        indexes = peakutils.indexes(y, thres=0.46, min_dist=30)
        # indexes = peakutils.gaussian_fit(x,y, center_only = False)
        # indexes = peakutils.interpolate(x,y, func=func(x,y))#, func= peakutils.gaussian_fit(x,y))
        # indexes = np.append(indexes, peakutils.indexes(-y, thres=0.45, min_dist=30))
        # print(indexes)
        # print(x[indexes], y[indexes])


        top_peaks_index = []
        top_peaks_values = []
        for i in indexes:
            if y[i] > 200:
                top_peaks_index.append(i)
                top_peaks_values.append(y[i])

        std = np.std(top_peaks_values)
        mean = np.median(top_peaks_values)

        outliers = [0]

        new_indexes = [0]
        for i in indexes:
            # testingvalue = y[i]
            # upper = std + (3 * mean)
            # lower = std - (3* mean)



            # if (y[i] < mean + (2 * std))   and y[i] > 200:#and (y[i] > mean - (4* std)) :#
            #     new_indexes.append(i)
            # elif y[i] > 200:
            #     outliers.append(i)

            if (y[i] < mean + (1 * std)) and (y[i] > mean - (1 * std)):  # and y[i] > 200:#
                new_indexes.append(i)
            elif y[i] > mean - (2 * std):
                outliers.append(i)

        new_indexes2 = [0]
        for i in new_indexes:
            down = max(0, i - 10)
            up = min(len(y) - 1, i + 10)
            if (y[down] < 0.2 * y[i]) and (y[up] < 0.2 * y[i]):
                new_indexes2.append(i)
                # elif y[i] > 200:
                #     outliers.append(i)

        indexes = new_indexes2

        new_indexes3 = [0]
        for i in range(0, len(new_indexes2) - 1):
            if (new_indexes2[i + 1] - new_indexes2[i] > 100):
                new_indexes3.append(new_indexes2[i])
            else:
                outliers.append(new_indexes2[i])

        indexes = new_indexes3
        indexes2 = copy.copy(outliers)
        indexes3 = copy.copy(indexes)

        outliers = sorted(outliers)

        it_out = iter(outliers)
        it_in = iter(indexes)

        val_it = next(it_in, None)
        val_out = next(it_out, None)

        new_indexes4 = []

        while (True):

            if (val_it is None or val_out is None):
                break
            if (val_it <= val_out and val_out >= 0):
                val_it_next = next(it_in, None)
                while (True):
                    if (val_it_next is None or val_out is None):
                        val_it = None
                        break
                    if (val_it_next > val_out):
                        break
                    if (val_it_next < val_out):
                        # new_indexes4.append(val_it_next)
                        val_it = val_it_next

                    val_it_next = next(it_in, None)
                if (val_it is None or val_out is None):
                    break

                if val_it == 0:
                    new_r = y[val_it_next]
                else:
                    new_r = (y[val_it] + y[val_it_next]) / 2

                y[val_it] = new_r

                # for i in range(val_it+1,val_it_next+1):
                y = np.delete(y, range(val_it + 1, val_it_next + 1))
                # y.remove(y[i])

                # new_indexes4.append(val_it)
                indexes.remove(val_it_next)

                difference = val_it_next - val_it
                # val_it = val_it_next
                for j in range(0, len(outliers)):
                    if (outliers[j] >= val_it):
                        outliers[j] -= difference

                del_index = []
                for k in range(0, len(indexes)):
                    if indexes[k] == val_it_next:
                        del_index.append(k)
                    elif (indexes[k] > val_it):
                        indexes[k] -= difference
                        # indexes = np.delete(indexes, del_index)
            val_out = next(it_out, None)

        length = len(y)
        x = np.linspace(0, length - 1, length)

        length = len(yold)
        x2 = np.linspace(0, length - 1, length)

        # indexes = signal.find_peaks_cwt(y, np.arange(1, 10))


        pyplot.close("all")

        pyplot.figure(figsize=(10, 6))
        pplot(x2, yold, indexes2)
        pyplot.title('outliers')

        pyplot.figure(figsize=(10, 6))
        pplot(x2, yold, indexes3)
        pyplot.title('original')

        pyplot.figure(figsize=(10, 6))
        pplot(x, y, indexes)
        pyplot.title('Cleaned')

        # pyplot.close("all")
        # pyplot.figure(figsize=(10, 6))
        # pplot(x2, yold, indexes3)
        # pyplot.title('original')

        print "hi"



#
# mat2 = scipy.io.loadmat('A00002.mat')
# mat3 = scipy.io.loadmat('A00003.mat')
# mat4 = scipy.io.loadmat('A00004.mat')
# mat5 = scipy.io.loadmat('A00005.mat')
# mat6 = scipy.io.loadmat('A00006.mat')


print "done"
