"""
 Created by Nidhi Mundra on 26/04/17.
"""

import cPickle
import gc
import os

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_reader import DataReader
from feature_generator import FeatureGenerator
from preprocessor import Preprocessor
from scorer import Scorer

data_reader = DataReader()
feature_generator = FeatureGenerator()
features = []
labels = []
file_names = []
preprocessor = Preprocessor()

for filename in os.listdir('training2017'):
    if filename.endswith('.mat'):
        data, label = data_reader.fetch(filename[:-4])
        try:
            data = preprocessor.process(data)
            features.append(feature_generator.get_features(data))
            labels.append(label)
            file_names.append(filename[:-4])
            print filename[:-4]
        except:
            print filename[:-4], "HAD AN ERROR!!"
             # TODO: Do normal classification here - use rolling mean, std, var, max, min, etc  - Nidhi

gc.disable()
with open('wave_features.pickle', 'wb') as handle:
    cPickle.dump(features, handle, protocol=cPickle.HIGHEST_PROTOCOL)
gc.enable()

gc.disable()
with open('wave_labels.pickle', 'wb') as handle:
    cPickle.dump(labels, handle, protocol=cPickle.HIGHEST_PROTOCOL)
gc.enable()

gc.disable()
with open("wave_filenames.pickle", 'wb') as handle:
    cPickle.dump(file_names, handle, protocol=cPickle.HIGHEST_PROTOCOL)
gc.enable()

gc.disable()
with open('wave_features.pickle', 'rb') as handle:
    X = cPickle.load(handle)
gc.enable()

gc.disable()
with open('wave_labels.pickle', 'rb') as handle:
    Y = cPickle.load(handle)
gc.enable()

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2)

feature_selector = SelectKBest(f_regression, k=5)

classifier = AdaBoostClassifier()
base_classifier = RandomForestClassifier()

params = {
    "base_estimator": [base_classifier],
    "n_estimators": range(30, 61, 10),
    "learning_rate": np.arange(0.8, 1.01, 0.05),
}

clf = GridSearchCV(classifier, param_grid=params, cv=10)

pipeline = Pipeline([('feature_selector', feature_selector), ('clf', clf)])
pipeline.fit(Xtr, Ytr)
prediction = pipeline.predict(Xte)
print "Accuracy: ", Scorer.score(Xte, Yte)


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
