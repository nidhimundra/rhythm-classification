"""
 Created by Nidhi Mundra on 25/04/17.
"""

import csv
import scipy.io


class DataReader:
    def __init__(self):
        self.path = "training2017/"
        self.label_filename = "REFERENCE.csv"
        self.read_labels()

    def read_labels(self):
        with open(self.path + self.label_filename, mode='r') as infile:
            reader = csv.reader(infile)
            self.labels = {rows[0]: rows[1] for rows in reader}

    def read_matlab_data(self, filename):
        mat1 = scipy.io.loadmat(self.path + filename)
        return mat1['val'][0]

    def fetch(self, filename):
        data = self.read_matlab_data(filename)
        label = self.labels[filename]
        return [data, label]
