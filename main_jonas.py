import csv
import os

import scipy.io

from preprocessor import Preprocessor
from r_peak_finder import RPeakFinder


def read_lable_dict():
    with open('training2017/REFERENCE.csv', mode='r') as infile:
        reader = csv.reader(infile)
        # with open('coors_new.csv', mode='w') as outfile:
        # writer = csv.writer(outfile)
        mydict = {rows[0]: rows[1] for rows in reader}
    return mydict


preprocesser = Preprocessor()
labels = []
features = []
for filename in os.listdir('training2017'):
    if filename.endswith('.mat'):
        name = filename[:-4]
        label_dict = read_lable_dict()
        label = label_dict[name]
        mat1 = scipy.io.loadmat('training2017/' + filename)
        # plot_line_graph([mat1['val'][0]], [label], name)
        data = mat1['val'][0]

        # rPeaks = RPeakFinder(data)
        # rPeaks.plot("original")
        # data = preprocesser.process(data)
        # rPeaks = RPeakFinder(data)
        # rPeaks.plot("preprocessed")
        # rPeaks = RPeakFinder(data)
        # rPeaks.plot("original")
        # data = preprocesser.process(data)
        # rPeaks = RPeakFinder(data)
        # rPeaks.plot("preprocessed")
        # # rPeaks.find_outliers()
        # rPeaks.r_detection_outlier_removal()
        try:
            rPeaks = RPeakFinder(data)
            rPeaks.plot("original")
            data = preprocesser.process(data)
            rPeaks = RPeakFinder(data)
            rPeaks.plot("preprocessed")
            # rPeaks.find_outliers()
            rPeaks.r_detection_outlier_removal()
            # rPeaks.remove_outliers()
            rPeaks.plot("Outliers removed")
            features.append(rPeaks.feature_generation())
            labels.append(label)
        except:
            print(filename)
            print "had an error \n"

# with open('features.pickle', 'wb') as handle:
#     pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('labels.pickle', 'wb') as handle:
#     pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('features.pickle', 'rb') as handle:
#     features = pickle.load(handle)
# with open('labels.pickle', 'rb') as handle:
#     labels = pickle.load(handle)
#
#
# new_labels = []
#
# new_features = []
# x=float('nan')
# for l in range(0, len(labels)):
#     if len(features[l]) == 120:
#
#         new_features.append(features[l])
#
#         if labels[l] == "N":
#             new_labels.append(0)
#         elif labels[l] == "O":
#             new_labels.append(1)
#         elif labels[l] == "A":
#             new_labels.append(2)
#         elif labels[l] == "~":
#             new_labels.append(3)
#         else:
#             print l
#     else:
#         print features[l]
#
# new_features = normalize(new_features)
# X_train, X_test, y_train, y_test = train_test_split(new_features, new_labels , test_size=0.2)
# #
#
# with open('X_train.pickle', 'wb') as handle:
#     pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('X_test.pickle', 'wb') as handle:
#     pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('y_train.pickle', 'wb') as handle:
#     pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('y_test.pickle', 'wb') as handle:
#     pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# with open('X_train.pickle', 'rb') as handle:
#     X_train = pickle.load(handle)
# with open('X_test.pickle', 'rb') as handle:
#     X_test = pickle.load(handle)
# with open('y_train.pickle', 'rb') as handle:
#     y_train = pickle.load(handle)
# with open('y_test.pickle', 'rb') as handle:
#     y_test = pickle.load(handle)
#
# classifier = AdaBoostClassifier()
# base_classifier = RandomForestClassifier()
#
# params = {
#     "base_estimator": [base_classifier],
#     "n_estimators": range(30, 61, 10),
#     "learning_rate": np.arange(0.8, 1.01, 0.05),
#     # "base_estimator__n_estimators": range(5,15,5),
#     # "base_estimator__criterion": ["gini", "entropy"],
#     # "base_estimator__max_features":
#
# }

# new_features = np.array(new_features)
# new_labels = np.array(new_labels)
#
# cv = GridSearchCV(classifier, param_grid=params, cv=10)
# cv.fit(X_train, y_train)
# prediction = cv.best_estimator_.predict(X_test)
# accuracy = accuracy_score(y_test, prediction)
#
# print cv.best_score_
