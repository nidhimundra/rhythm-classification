"""
 Created by Nidhi Mundra on 26/04/17.
"""

import pandas as pd
import numpy as np
import math


class BasicPeakFinder:
    def __init__(self, data):
        # Initialize constants
        self.HRW = 6.75
        self.FS = 300

        # Initialize data
        self.data = data

        # Initialize R Peaks array
        self.r_peaks = []

        # Computing Rolling Means in the data
        self.__roll_mean__()

        # Detect Peaks using the Rolling Means
        self.__detect_peaks__()

    def __roll_mean__(self):
        """
        Computing Rolling Mean of windows in the data
        """

        self.rolling_mean = pd.rolling_mean(self.data, window=int(self.HRW * self.FS))

        # Assign average heartbeat rate to NaNs
        avg_hr = (np.mean(self.data))
        self.rolling_mean = [avg_hr if math.isnan(x) else x for x in self.rolling_mean]

        # Increase the threshold of Moving Average
        self.rolling_mean = [x * 15 for x in self.rolling_mean]

    def __detect_peaks__(self):
        """
        Detect the R Peaks using the Rolling Means in the windows
        """

        window = []
        list_position = 0

        for data_point in self.data:
            # Get mean of current data point
            mean = self.rolling_mean[list_position]

            # If the data point falls below mean, move to the next data point
            if (data_point <= mean) and (len(window) <= 1):
                list_position += 1

            # If the data point falls above mean, add the data point in the window and move to the next data point
            elif data_point > mean:
                window.append(data_point)
                list_position += 1

            # Otherwise, create next window
            else:
                beat_position = list_position - len(window) + (window.index(max(window)))
                self.r_peaks.append(beat_position)
                window = []
                list_position += 1

    def get_peaks_data(self):
        """
        :return: R Peaks and ECG wave data 
        """
        return [self.r_peaks, self.data]
