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
Xtr, Ytr, filenames = data_reader.fetch_data_and_labels(path='training_data')
clf.fit(Xtr, Ytr, filenames)

# Predict the accuracy score of the test output
Xte, Yte = data_reader.fetch_data_and_labels(path='testing_data')
print "Accuracy: ", clf.score(Xte, Yte)

# TODO: Classify waves whose peaks were unidentified - Nidhi
# TODO: Test Accuracy Scorer - Nidhi

# TODO: Intermediate Peak Finding - Jonas
# TODO: Feature Selection
# TODO: Dimensionality Reduction
# TODO: Classification Model Selection and their Hyperparameter Optimization
# TODO: Heatmap for correlation between features - Nidhi - Not on Priority
# TODO: Eliminate DeprecationWarning for 1d arrays