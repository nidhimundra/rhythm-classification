import pandas as pd
import numpy as np
import math
from scipy.io import loadmat
from matplotlib import pyplot
from peakutils.plot import plot as pplot


class HeartBeat:
    def __init__(self, path, filename):
        self.hrw = 2.25
        self.fs = 300
        self.measures = dict()
        self.path = path
        self.filename = filename
        self.dataset = self.get_data()

    def get_data(self):
        mat = loadmat(self.path + self.filename)
        return pd.DataFrame({'val': mat['val'][0]})

    def roll_mean(self):
        mov_avg = pd.rolling_mean(self.dataset.val, window=int(self.hrw * self.fs))
        avg_hr = (np.mean(self.dataset.val))
        mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
        mov_avg = [x * 1.2 for x in mov_avg]
        self.dataset['rollingmean'] = mov_avg

    def detect_peaks(self):
        window = []
        peaklist = []
        listpos = 0
        for datapoint in self.dataset.val:
            rollingmean = self.dataset.rollingmean[listpos]
            if (datapoint <= rollingmean) and (len(window) <= 1):
                listpos += 1
            elif datapoint > rollingmean:
                window.append(datapoint)
                listpos += 1
            else:
                beatposition = listpos - len(window) + (window.index(max(window)))
                peaklist.append(beatposition)
                window = []
                listpos += 1
        self.measures['peaklist'] = peaklist
        self.measures['ybeat'] = [self.dataset.val[x] for x in peaklist]

    def calc_RR(self):
        RR_list = []
        peaklist = self.measures['peaklist']
        cnt = 0
        while (len(peaklist) - 1) > cnt:
            RR_interval = (peaklist[cnt + 1] - peaklist[cnt])
            ms_dist = ((RR_interval / self.fs) * 1000.0)
            RR_list.append(ms_dist)
            cnt += 1
        self.measures['RR_list'] = RR_list

    def calc_bpm(self):
        RR_list = self.measures['RR_list']
        self.measures['bpm'] = 60000 / np.mean(RR_list)

    def plot(self):
        peaklist = self.measures['peaklist']
        ybeat = self.measures['ybeat']
        pyplot.close("all")
        pyplot.figure(figsize=(10, 6))
        length = len(self.dataset.val)
        x = np.linspace(0, length - 1, length)
        pplot(x, self.dataset.val, peaklist)
        pyplot.title("HeartBeat")
        # pyplot.plot(mov_avg, color='green')  # Plot moving average
        pyplot.show()

    def process(self):
        self.roll_mean()
        self.detect_peaks()
        self.calc_RR()
        self.calc_bpm()
        self.plot()
        print self.measures
