from heartbeat import HeartBeat
import os

path = 'training2017/'
for filename in os.listdir(path):
    if filename.endswith('.mat'):
        hb = HeartBeat(path, filename)
        hb.process()
