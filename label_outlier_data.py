import csv
import os
import pickle

import numpy as np
import scipy.io
from matplotlib import pyplot

from peakutils.plot import plot as pplot


def read_lable_dict():
    with open('training2017/REFERENCE.csv', mode='r') as infile:
        reader = csv.reader(infile)
        # with open('coors_new.csv', mode='w') as outfile:
        # writer = csv.writer(outfile)
        mydict = {rows[0]: rows[1] for rows in reader}
    return mydict


# def read_all_lable_dict():
#     with open('training2017/REFERENCE.csv', mode='r') as infile:
#         reader = csv.reader(infile)
#         # with open('coors_new.csv', mode='w') as outfile:
#         # writer = csv.writer(outfile)
#         lower = {}
#
#         lower["label"] = rows[1]
#
#         mydict = {rows[0]: {"label" : rows[1], "flip": rows[2],} for rows in reader}
#     return mydict

# all_labels = {}
with open('all_labels.pickle', 'rb') as handle:
    all_labels = pickle.load(handle)
label_dict = read_lable_dict()

dir = 'training_data'
for filename in os.listdir(dir):
    if filename.endswith('.mat'):
        name = filename[:-4]

        if (name not in all_labels):
            # all_labels["A01096"]["right"] = 0
            # print filename
            label = label_dict[name]
            mat1 = scipy.io.loadmat('training_data/' + filename)
            # plot_line_graph([mat1['val'][0]], [label], name)
            y = mat1['val'][0]
            length = len(y)
            x = np.linspace(0, length - 1, length)

            pyplot.close("all")

            pyplot.figure(figsize=(10, 6))
            pplot(x, y, [0])
            pyplot.title('outliers')
            pyplot.show()

            var = raw_input("Please enter something: ")
            print "you entered", var

            var = var.split(",")

            all_labels[name] = {}
            all_labels[name]["flip"] = var[0]
            all_labels[name]["left"] = var[1]
            all_labels[name]["middle"] = var[2]
            all_labels[name]["right"] = var[3]

            all_labels[name]["label"] = label

            with open('all_labels.pickle', 'wb') as handle:
                pickle.dump(all_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
