"""
 Created by Nidhi Mundra on 26/04/17.
"""

import numpy as np


class Scorer:
    def __init__(self):
        """
        Initializes data structures needed to compute score
        """

        # List of all the class labels
        self.labels = [0, 1, 2, 3]

        # Dictionary to store count of each label in predicted labels list
        self.total_prediction_count = {0: 0, 1: 0, 2: 0, 3: 0}

        # Dictionary to store count of each label in actual labels list
        self.total_actual_count = {0: 0, 1: 0, 2: 0, 3: 0}

        # Dictionary to store count of correctly predicted labels
        self.total_correct_prediction_count = {0: 0, 1: 0, 2: 0, 3: 0}

    def __get_f1_scores__(self):
        """
        Compute F1 score of each label using the formula given at https://www.physionet.org/challenge/2017/
        
        :return: Numpy array of F1 scores of each label 
        """

        # Initialize empty output array
        output = np.array([])

        for label in self.labels:
            # Compute f1 value for current label

            if self.total_correct_prediction_count[label] != 0:
                f1 = float(2 * self.total_correct_prediction_count[label]) / \
                     float(self.total_actual_count[label] + self.total_prediction_count[label])
            else:
                f1 = 0.0

            # Append computed f1 score to output array
            output = np.append(output, f1)

        return output

    def score(self, predicted_y, actual_y):
        """
        Compute the classification score based on predicted and actual labels 
        Formula given at https://www.physionet.org/challenge/2017/
        
        :param predicted_y: Array containing all the predicted labels
        
        :param actual_y: Array containing all the actual labels
        
        :return: Prediction Score 
        """

        self.labels = [0, 1, 2, 3]

        # Dictionary to store count of each label in predicted labels list
        self.total_prediction_count = {0: 0, 1: 0, 2: 0, 3: 0}

        # Dictionary to store count of each label in actual labels list
        self.total_actual_count = {0: 0, 1: 0, 2: 0, 3: 0}

        # Dictionary to store count of correctly predicted labels
        self.total_correct_prediction_count = {0: 0, 1: 0, 2: 0, 3: 0}

        for i in range(0, len(predicted_y)):
            # Extract predicted and actual labels for ith record
            predicted_label = predicted_y[i]
            actual_label = actual_y[i]

            # Increment the count of corrected predicted label if predicted and actual labels are same
            if predicted_label == actual_label:
                self.total_correct_prediction_count[actual_label] += 1

            # Increment total counts
            self.total_actual_count[actual_label] += 1
            self.total_prediction_count[predicted_label] += 1

        # Compute f1 scores of each label and return their mean
        return np.mean(self.__get_f1_scores__())
