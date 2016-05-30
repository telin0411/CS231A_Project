import cv2
import csv
import numpy as np
from numpy import genfromtxt

path = '/mnt0/data/img-cap/models/model_id3_data.csv'

sim_matrix = genfromtxt(path, delimiter=',')
min_val = np.amin(sim_matrix)
max_val = np.amax(sim_matrix)
sim_matrix = 255*(sim_matrix-min_val)/(max_val-min_val)
sim_matrix = sim_matrix.astype('uint8')
#sim_matrix = cv2.equalizeHist(sim_matrix)
#sim_matrix = cv2.GaussianBlur(sim_matrix, (5,5), 0)
sim_matrix_color = cv2.applyColorMap(sim_matrix, cv2.COLORMAP_JET)

cv2.namedWindow('sim', cv2.WINDOW_NORMAL)
cv2.imshow('sim', sim_matrix_color)
cv2.waitKey(0)
