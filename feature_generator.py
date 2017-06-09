"""
 Created by Nidhi Mundra on 25/04/17.
"""

import copy

import numpy as np
from scipy.stats import skew
from sklearn.linear_model import LinearRegression

from peak_finder import PeakFinder
from rolling_stats_calculator import RollingStatsCalculator


class FeatureGenerator:
    def __init__(self):
        # Initialize stats calculator
        self.rolling_stats_calculator = RollingStatsCalculator()

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

        for i in range(0, len(feature_matrix)):
            for j in range(0, len(feature_matrix[0])):
                try:
                    feature_matrix[i][j] = float(feature_matrix[i][j])
                except:
                    pass

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

    def __inner_peak_picking__(self, distance, data):
        data = np.array(data)
        old_data = copy.deepcopy(data)
        data, first_min, last_min = self.__delete_r_peaks(data)
        left_max, right_max = self.__get_inbetween_peaks__(data)

        if first_min == None:
            first_min = 0
        if last_min == None:
            last_min = len(data) - 1
        if left_max == None:
            left_max = 0
        if right_max == None:
            right_max = len(data) - 1

        # peaks = peakutils.indexes(data, thres=0.46, min_dist=30)
        # peak_finder.plot("intermediate", old_data, [0])
        peaks = np.array([0, int(first_min), int(left_max + first_min), int(right_max + first_min), int(last_min),
                          len(old_data) - 1])
        # peak_finder.plot("intermediate", old_data, peaks)


        rstpqr_features = self.__get_rstpqr_features(old_data, peaks)
        return rstpqr_features

        # print "here"

    def __get_rstpqr_features(self, data, peaks):
        features = []
        for i in range(0, len(peaks) - 1):
            features = np.append(features, self.__generate_peakwise_rstpqr_features__(data, peaks[i], peaks[i + 1]))
        return features

    def __generate_peakwise_rstpqr_features__(self, data, left, right):
        features = []
        interdata = data[left: right]
        features = np.append(features, [data[left], data[right]])  # the value of the peaks on the y axis
        features = np.append(features, right - left)  # distance between the two peaks
        features = np.append(features, abs(data[left] - data[right]))  # difference in height between the two peaks
        # if (left == right):
        #     print ""
        extrema_from_left = self.__get_next_extrema__(data, left, forward=True)
        extrema_from_right = self.__get_next_extrema__(data, right, forward=False)

        if (None in features) | (extrema_from_right == None) | (extrema_from_left == None):
            features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        else:
            features = np.append(features, [data[extrema_from_left], data[extrema_from_right]])
            features = np.append(features, left - extrema_from_left)
            features = np.append(features, extrema_from_left - extrema_from_right)
            features = np.append(features, extrema_from_right - right)
            features = np.append(features, np.std(interdata))
            features = np.append(features, np.mean(interdata))
            lr = LinearRegression()
            try:
                lr.fit(np.array(list(range(0, len(interdata)))).reshape((len(interdata), 1)),
                       interdata.reshape((len(interdata), 1)))
                # params = lr.get_params(deep=True)
                features = np.append(features, lr.coef_)
                features = np.append(features, lr.residues_)
            except:
                features = np.append(features, 0.0)
                features = np.append(features, 0.0)

        # if (None in features) | (len(features) != 13) | (extrema_from_right == None) | (extrema_from_left == None) :
        #     features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if len(features) != 13:
            features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        return features

    def __extrema__(self, data, point):
        if (point == 0):
            return self.__right_extrema__(data, point)
        elif (point == len(data) - 1):
            return self.__left_extrema__(data, len(data) - 1)
        else:
            left = self.__left_extrema__(data, point)
            right = self.__right_extrema__(data, point)
            if left == right:
                return left
            else:
                if left == None:
                    left = ""
                if right == None:
                    right = ""
                return left + right

    def __right_extrema__(self, data, point):

        i = point + 1
        while i < len(data):
            if data[point] < data[i]:
                return "MAX"
            elif data[point] > data[i]:
                return "MIN"
            i += 1
        return "MAX"

    def __left_extrema__(self, data, point):

        i = point - 1
        while i >= 0:
            if data[point] < data[i]:
                return "MAX"
            elif data[point] > data[i]:
                return "MIN"
            i -= 1

    def __get_next_extrema__(self, data, point, forward):

        extrema = self.__extrema__(data, point)

        if forward is True:
            iterator = list(range(point, len(data)))
            if extrema == "MINMAX":
                extrema = "MAX"
            elif extrema == "MAXMIN":
                extrema = "MIN"
        else:
            iterator = list(range(point, 0, -1))
            if extrema == "MINMAX":
                extrema = "MIN"
            elif extrema == "MAXMIN":
                extrema = "MAX"

        if extrema == "MAX":
            maximum = data[point]
            for i in iterator:
                if maximum < data[i]:
                    if forward is True:
                        return i - 1
                    return i + 1
                maximum = data[i]

        elif extrema == "MIN":
            minimum = data[point]
            for i in iterator:
                if minimum > data[i]:
                    if forward is True:
                        return i - 1
                    return i + 1
                minimum = data[i]

    def __get_inbetween_peaks__(self, data):
        length = len(data)
        left_max = self.__inbetween_peak__(data, 0, int(length / 2))
        right_max = self.__inbetween_peak__(data, int(length / 2), length)
        return left_max, right_max

    def __inbetween_peak__(self, data, left, right):

        argmax = 0
        max = -999999
        current_max = -999999
        for i in range(left, right):
            if data[i] >= current_max:
                current_max = data[i]
            elif data[i] < current_max:
                if current_max > max:
                    max = current_max
                    argmax = i - 1
                current_max = data[i]
        if argmax == 0:
            return int(left + (right - left) / 2)
        return argmax

    def __delete_r_peaks(self, data):
        first_min = self.__find_first_min__(data)
        last_min = self.__find_last_min__(data)
        return data[first_min:last_min], first_min, last_min

    def __find_first_min__(self, data):
        current_min = 999999999999
        for i in range(0, len(data)):
            if data[i] <= current_min:
                current_min = data[i]
            else:
                return i - 1

    def __find_last_min__(self, data):
        current_min = 999999999999
        for i in range(-len(data) + 1, 0):
            if data[-i] <= current_min:
                current_min = data[-i]
            else:
                return -i + 1




    def __get_wavelet_features__(self, distance, data):

        """
        Get features of each wavelet
        
        :param distance: Distance between current R-R peak
         
        :param data: Data points
        
        :return: An array containting all the features of current wavelet
        """

        # Initialize output array
        inner_features = []
        inner_features = self.__inner_peak_picking__(distance, data).tolist()
        # Compute distances between the intermediate peaks
        peak_distances = self.__get_intermediate_peak_distances__(distance, data)

        for j in range(0, len(peak_distances) - 1):

            # Find data between adjacent peaks
            sub_range = list(range(peak_distances[j], peak_distances[j + 1]))
            sub_data = [data[k] for k in sub_range]

            # Append features if the array contains some data points, append zeros otherwise
            if len(sub_data) != 0:

                # Compute wavelet features
                crest = np.nanmax(sub_data)
                trough = np.nanmin(sub_data)
                mean = np.nanmean(sub_data)
                std = np.nanstd(sub_data)
                wavelength = distance
                wave_height = crest - trough
                rms = np.sqrt(np.nanmean([i ** 2 for i in sub_data]))
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

        # k = 3
        # if len(feature_matrix) > k:
        #     estimator = KMeans(n_clusters=k)
        #     estimator.fit(feature_matrix)
        #     features = estimator.cluster_centers_.flatten()

        # Compute features based on the r_peaks and distances
        features = np.append(features, np.max(r_peaks))
        features = np.append(features, np.min(r_peaks))
        features = np.append(features, np.mean(r_peaks))
        features = np.append(features, np.std(r_peaks))
        features = np.append(features, np.max(distances))
        features = np.append(features, np.min(distances))
        features = np.append(features, np.mean(distances))
        features = np.append(features, np.std(distances))

        # Compute overall wave features and appended them in the main array
        feature_matrix = np.array(feature_matrix)
        for i in range(0, len(feature_matrix[0])):
            features = np.append(features, np.max(feature_matrix[:, i]))
            features = np.append(features, np.min(feature_matrix[:, i]))
            features = np.append(features, np.mean(feature_matrix[:, i]))
            features = np.append(features, np.std(feature_matrix[:, i]))

        return features

    def get_features(self, data, outliers):
        """
        Get features of ECG wave
         
        :param data: ECG wave data points
         
        :return: generated features of the wave
        """
        # Get peaks and data of the given wave
        # self.r_peaks, self.data = BasicPeakFinder(data).get_peaks_data()

        peakfinder = PeakFinder(data, outliers)
        # peakfinder.plot("preprocessed")
        self.r_peaks, self.data = peakfinder.get_peaks_data()
        # peakfinder.peak_plot("All Peaks")
        # peakfinder.plot("rpeaks")

        """
        If the number of peaks are sufficient, then generate peak features
        Else, create features from the wave data
        """
        if len(self.r_peaks) >= 10:
            # Get features from the extracted peaks and data
            return [self.__generate_features__(), "peak"]
        else:
            self.data = data[100:500]
            # Get features from the data
            return [self.__generate_rolling_features__(), "points"]

    def __generate_rolling_features__(self):

        """
        Generate rolling features of data
        :return: Array containing all the features
        """
        return self.rolling_stats_calculator.rolling_mean(self.data) \
               + self.rolling_stats_calculator.rolling_kurt(self.data) \
               + self.rolling_stats_calculator.rolling_max(self.data) \
               + self.rolling_stats_calculator.rolling_min(self.data) \
               + self.rolling_stats_calculator.rolling_sum(self.data) \
               + self.rolling_stats_calculator.rolling_var(self.data) \
               + self.rolling_stats_calculator.rolling_std(self.data) \
               + self.rolling_stats_calculator.rolling_skew(self.data) \
               + self.rolling_stats_calculator.rolling_count(self.data)
