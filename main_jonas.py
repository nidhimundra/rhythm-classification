import csv
import os

import scipy.io

from r_peak_finder import RPeakFinder


def read_lable_dict():
    with open('training2017/REFERENCE.csv', mode='r') as infile:
        reader = csv.reader(infile)
        # with open('coors_new.csv', mode='w') as outfile:
        # writer = csv.writer(outfile)
        mydict = {rows[0]: rows[1] for rows in reader}
    return mydict


for filename in os.listdir('training2017'):
    if filename.endswith('.mat'):
        name = filename[:-4]
        label_dict = read_lable_dict()
        label = label_dict[name]
        mat1 = scipy.io.loadmat('training2017/' + filename)
        # plot_line_graph([mat1['val'][0]], [label], name)
        data = mat1['val'][0]

        rPeaks = RPeakFinder(data)
        rPeaks.plot("original")
        # rPeaks.find_outliers()
        rPeaks.r_detection_outlier_removal()
        # rPeaks.remove_outliers()
        rPeaks.plot("Outliers removed")

        print(filename)
