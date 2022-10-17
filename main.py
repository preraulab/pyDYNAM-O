import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, THRESH_BINARY_INV, THRESH_OTSU, threshold
from matplotlib import pyplot as plt
img = imread('water_coins.jpg')
gray = cvtColor(img, COLOR_BGR2GRAY)
ret, thresh = threshold(gray, 0, 255, THRESH_BINARY_INV+THRESH_OTSU)
