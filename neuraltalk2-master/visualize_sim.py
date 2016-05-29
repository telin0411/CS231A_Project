import cv2
import csv
import numpy as np
from numpy import genfromtxt

path = '/mnt0/data/img-cap/models/model_id3_data.csv'

sim_matrix = genfromtxt(path, delimiter=',')
min_val = np.amin(sim_matrix)
max_val = np.amax(sim_matrix)
sim_matrix = (sim_matrix-min_val)/(max_val-min_val)

cv2.namedWindow('sim', cv2.WINDOW_NORMAL)
cv2.imshow('sim', sim_matrix)
cv2.waitKey(0)
