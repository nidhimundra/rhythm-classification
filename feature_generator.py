"""
 Created by Nidhi Mundra on 25/04/17.
"""

import numpy as np
from scipy.stats import skew
from sklearn.cluster import KMeans

from peak_finder import PeakFinder
from basic_peak_finder import BasicPeakFinder


class FeatureGenerator:
    def __init__(self):
        pass

    def __get_data_between_peaks__(self, first_peak, second_peak):
        """
        Get wave data between two peaks, Peak 1 and Peak 2
        :param first_peak: Peak 1
        :param second_peak: Peak 2
        :return: Wave data points
        """

        # Initialize output array
        data = []

        # Append data points between first and second peak into the output array
        for i in range(first_peak, second_peak + 1):
            data.append(self.data[i])

        return data

    def __generate_features__(self):
        """
        Generate all the features from the R Peaks and ECG wave data
        :return: Extracted features
        """

        # Initialize output arrays
        feature_matrix = []
        distances = []
        r_peaks = []

        # Find features for each wavelet
        for i in range(0, len(self.r_peaks) - 1):
            # Distance and data points between adjacent peaks
            distance = self.r_peaks[i + 1] - self.r_peaks[i]
            new_data = self.__get_data_between_peaks__(self.r_peaks[i], self.r_peaks[i + 1])

            distances.append(distance)
            r_peaks.append(new_data[0])

            # Extract features of wavelet
            feature_matrix.append(self.__get_wavelet_features__(distance, new_data))

        # Combine wavelet features to generate features of the whole wave
        return self.__get_wave_features__(feature_matrix, r_peaks, distances)

    def __get_intermediate_peak_distances__(self, distance, data):
        """
        Return distances between P, Q, R, S and T points of the ECG wave
        :param distance: distance between R-R peak
        :param data: Data points
        :return: An array containting all the intermediate distances
        """

        # Distance between intermediate peaks of the ECG wave
        s_len = int(round((1.0 / 13) * distance))
        st_len = int(round((2.0 / 13) * distance)) + s_len
        t_len = int(round((4.0 / 13) * distance)) + st_len
        u_len = int(round((1.0 / 13) * distance)) + t_len
        p_len = int(round((2.0 / 13) * distance)) + u_len
        pr_len = int(round((2.0 / 13) * distance)) + p_len
        q_len = len(data)

        return [0, s_len, st_len, t_len, u_len, p_len, pr_len, q_len]

    def __get_wavelet_features__(self, distance, data):

        """
        Get features of each wavelet
        :param distance: Distance between current R-R peak 
        :param data: Data points
        :return: An array containting all the features of current wavelet
        """

        # Initialize output array
        inner_features = []

        # Compute distances between the intermediate peaks
        peak_distances = self.__get_intermediate_peak_distances__(distance, data)

        for j in range(0, len(peak_distances) - 1):

            # Find data between adjacent peaks
            sub_range = range(peak_distances[j], peak_distances[j + 1])
            sub_data = [data[k] for k in sub_range]

            # Append features if the array contains some data points, append zeros otherwise
            if len(sub_data) != 0:

                # Compute wavelet features
                crest = np.max(sub_data)
                trough = np.min(sub_data)
                mean = np.mean(sub_data)
                std = np.std(sub_data)
                wavelength = distance
                wave_height = crest - trough
                rms = np.sqrt(np.mean([i ** 2 for i in sub_data]))
                skewness = skew(sub_data)

                inner_features.append(crest)
                inner_features.append(trough)
                inner_features.append(mean)
                inner_features.append(std)
                inner_features.append(wavelength)
                inner_features.append(wave_height)
                inner_features.append(rms)
                inner_features.append(skewness)
            else:
                inner_features.append(0)
                inner_features.append(0)
                inner_features.append(0)
                inner_features.append(0)
                inner_features.append(0)
                inner_features.append(0)
                inner_features.append(0)
                inner_features.append(0)

        return inner_features

    def __get_wave_features__(self, feature_matrix, r_peaks, distances):
        """
        Extract features of the whole wave from the wavelet features
        :param feature_matrix: 2D array contatining features of each wavelet
        :param r_peaks: R Peak points
        :param distances: Distance between R-R peak
        :return: Features array of the whole wave 
        """

        # Initialize output array
        features = []

        # Find different clusters of the wavelet features
        # Append the cluster means to the main features array

        k = 3
        if len(feature_matrix) > k:
            estimator = KMeans(n_clusters=k)
            estimator.fit(feature_matrix)
            features = estimator.cluster_centers_.flatten()

        # Compute features based on the r_peaks and distances
        features = np.append(features, np.max(r_peaks))
        features = np.append(features, np.min(r_peaks))
        features = np.append(features, np.mean(r_peaks))
        features = np.append(features, np.std(r_peaks))
        features = np.append(features, np.max(distances))
        features = np.append(features, np.min(distances))
        features = np.append(features, np.mean(distances))
        features = np.append(features, np.std(distances))


        # Compute overall wave features and appened them in the main array
        feature_matrix = np.array(feature_matrix)
        for i in xrange(len(feature_matrix[0])):
            features = np.append(features, np.max(feature_matrix[:, i]))
            features = np.append(features, np.min(feature_matrix[:, i]))
            features = np.append(features, np.mean(feature_matrix[:, i]))
            features = np.append(features, np.std(feature_matrix[:, i]))

        return features

    def get_features(self, data):
        """
        Get features of ECG wave 
        :param data: ECG wave data points 
        :return: generated features of the wave
        """
        # Get peaks and data of the given wave
        # self.r_peaks, self.data = BasicPeakFinder(data).get_peaks_data()
        self.r_peaks, self.data = PeakFinder(data).get_peaks_data()

        # Get features from the extracted peaks and data
        return self.__generate_features__()
