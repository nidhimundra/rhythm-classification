"""
 Created by Nidhi Mundra on 25/04/17.
"""

import os
import csv
import scipy.io


class DataReader:
    def __init__(self):
        """
        Initialize the path, label file and labels
        """

        # Path of the data file
        self.path = ""

        # Label file
        self.label_filename = "REFERENCE.csv"

        # Labels map
        self.labels_map = {'N': 0, 'O': 1, 'A': 2, '~': 3}

        # Read and store the labels
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
        
        :return: Data, labels and file names of all the waves  
        """

        self.path = path + "/"

        # Initializing output arrays
        waves_data = []
        labels = []
        file_names = []

        count = 0
        for filename in os.listdir(path):

            # Fetch data from all .mat files
            if filename.endswith('.mat'):
                data, label = self.fetch(filename[:-4])

                # Append results to output arrays
                waves_data.append(data)
                labels.append(self.labels_map[label])
                file_names.append(filename[:-4])

            count += 1
            if count > 30:
                break

        return [waves_data, labels, file_names]
