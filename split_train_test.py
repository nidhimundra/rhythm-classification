"""
 Created by Jonas Pfeiffer on 26/04/17.
"""

import os
from random import randint
from shutil import copyfile

dir = 'training2017'
test_dir = 'testing_data'
train_dir = 'training_data'

# Splits the data in 80:20 files

for filename in os.listdir(dir):
    if filename.endswith('.mat'):
        mat = filename
        hea = filename.split(".")[0] + ".hea"

        if randint(0, 9) < 8:
            copyfile(dir + "/" + mat, train_dir + "/" + mat)
            copyfile(dir + "/" + hea, train_dir + "/" + hea)

        else:
            copyfile(dir + "/" + mat, test_dir + "/" + mat)
            copyfile(dir + "/" + hea, test_dir + "/" + hea)
