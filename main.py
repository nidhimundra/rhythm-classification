"""
 Created by Nidhi Mundra on 26/04/17.
"""

from data_reader import DataReader
from ecg_classifier import ECGClassifier

# Initialize data reader and classifier objects
data_reader = DataReader()
clf = ECGClassifier()

# Fit the training data in the classifier
Xtr, Ytr = data_reader.fetch_data_and_labels(path='training_data')
clf.fit(Xtr, Ytr)

# Predict the accuracy score of the test output
Xte, Yte = data_reader.fetch_data_and_labels(path='testing_data')
print "Accuracy: ", clf.score(Xte, Yte)

# TODO: Intermediate Peak Finding - Jonas
# TODO: Middle Outlier Elimination improvement - Hyperparameter optimization for preprocessor - Jonas
# TODO: Add outliers of Preprocessing to overall outliers.
# TODO: Code Refactoring - feature_generator, peak_finder, basic_peak_finder - Nidhi
# TODO: Code Refactoring - preprocessor, split_train_test - Jonas
# TODO: Classify waves whose peaks were unidentified - Nidhi
# TODO: Add accuracy scorer in GridSearchCV - Nidhi
# TODO: Test Accuracy Scorer - Nidhi
# TODO: Feature Selection
# TODO: Dimensionality Reduction
# TODO: Classification Model Selection and their Hyperparameter Optimization
# TODO: Heatmap for correlation between features - Nidhi - Not on Priority
