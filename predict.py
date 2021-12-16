import cv2
from sklearn import svm
import os
import numpy as np
import joblib
from skimage.feature import hog
import argparse

parser = argparse.ArgumentParser(description='Predict the image')
parser.add_argument('--image', type=str, help='Path to the image')

args = parser.parse_args()

imgfile = args.image

clf = joblib.load('person_final.pkl')

def crop_image(img):
    h, w, d = img.shape
    l = (w - 64) / 2
    t = (h - 128) / 2
    crop = img[int(t):int(t+128), int(l):int(l+64)]
    return crop

def read_image(imgfile):
    print("Reading images...")
    img_features = []
    # global total_img
    img = cv2.imread(imgfile)
    cropped = crop_image(img)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=9, pixels_per_cell=(
        8, 8), cells_per_block=(2, 2), block_norm="L2", feature_vector=True)
    img_features.append(features)

    return img_features

img_features = read_image(imgfile)
img_result = clf.predict(img_features)
if img_result == 1:
    print("It is a person")
else:
    print("It is not a person")