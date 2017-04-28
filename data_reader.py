"""
 Created by Nidhi Mundra on 25/04/17.
"""

import os
import csv
import scipy.io


class DataReader:
    def __init__(self):
        self.path = ""
        self.label_filename = "REFERENCE.csv"
        self.__read_labels__()

    def __read_labels__(self):
        """
        Read all the wave labels from the reference file
        Store all the labels in a map of filename and label
        """
        with open(self.path + self.label_filename, mode='r') as infile:
            reader = csv.reader(infile)
            self.labels = {rows[0]: rows[1] for rows in reader}

    def __read_matlab_data__(self, filename):
        """
        Read data from matlab file
        :param filename: Matlab file name
        :return: Data 
        """
        mat1 = scipy.io.loadmat(self.path + filename)
        return mat1['val'][0]

    def fetch(self, filename):
        """
        Fetch data from file
        :param filename: Name of the file
        :return: Data and label 
        """
        data = self.__read_matlab_data__(filename)
        label = self.labels[filename]
        return [data, label]

    def fetch_data_and_labels(self, path):
        """
        Fetch data and labels of each file in the path
        :param path: path from where the data needs to be fetched
        :return: Data and labels of all the waves  
        """

        self.path = path + "/"

        # Initializing output arrays
        waves_data = []
        labels = []
        filenames = []

        for filename in os.listdir(path):

            # Fetch data from all .mat files
            if filename.endswith('.mat'):
                data, label = self.fetch(filename[:-4])

                # Append results to output arrays
                waves_data.append(data)
                labels.append(label)
                filenames.append(filename[:-4])

        return [waves_data, labels, filenames]
