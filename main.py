"""
 Created by Nidhi Mundra on 26/04/17.
"""

import warnings

from data_reader import DataReader
from ecg_classifier import ECGClassifier

warnings.filterwarnings("ignore")

# Initialize data reader and classifier objects
data_reader = DataReader()
clf = ECGClassifier()

# Fit the training data in the classifier
Xtr, Ytr, file_names = data_reader.fetch_data_and_labels(path='training_data')
clf.fit(Xtr, Ytr, file_names)

# Predict the accuracy score of the test output
Xte, Yte, file_names = data_reader.fetch_data_and_labels(path='testing_data')
print "Accuracy: ", clf.score(Xte, Yte, file_names)
