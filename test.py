import cv2
from sklearn import svm
import os
import numpy as np
import joblib
from skimage.feature import hog
import argparse

parser = argparse.ArgumentParser(description='Parse Training Directory')
parser.add_argument('--pos', help='Path to directory containing Positive Images')
parser.add_argument('--neg', help='Path to directory containing Negative images')

args = parser.parse_args()

pos_img_dir = args.pos
neg_img_dir = args.neg

clf = joblib.load('person_final.pkl')

total_pos_samples = 0
total_neg_samples = 0

def crop_centre(img):
    h, w, d = img.shape
    l = (w - 64)/2
    t = (h - 128)/2
    #print (h, w, l, t)
    crop = img[int(t):int(t+128), int(l):int(l+64)]
    return crop


def read_filenames():

    f_pos = []
    f_neg = []

    for (dirpath, dirnames, filenames) in os.walk(pos_img_dir):
        f_pos.extend(filenames)
        break

    for (dirpath, dirnames, filenames) in os.walk(neg_img_dir):
        f_neg.extend(filenames)
        break

    print("Positive Image Samples: " + str(len(f_pos)))
    print("Negative Image Samples: " + str(len(f_neg)))

    return f_pos, f_neg

def read_images(f_pos, f_neg):

    print ("Reading Images")

    array_pos_features = []
    array_neg_features = []
    global total_pos_samples
    global total_neg_samples
    # print("Positive Samples: " + str(len(f_pos)))
    for imgfile in f_pos:
        img = cv2.imread(os.path.join(pos_img_dir, imgfile))
        cropped = crop_centre(img)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", feature_vector=True)
        array_pos_features.append(features.tolist())
        total_pos_samples += 1

    # print("Negative Samples: " + str(len(f_neg)))
    for imgfile in f_neg:
        img = cv2.imread(os.path.join(neg_img_dir, imgfile))
        cropped = crop_centre(img)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", feature_vector=True)
        array_neg_features.append(features.tolist())
        total_neg_samples += 1

    return array_pos_features, array_neg_features



pos_img_files, neg_img_files = read_filenames()

pos_features, neg_features = read_images(pos_img_files, neg_img_files)

pos_result = clf.predict(pos_features)
neg_result = clf.predict(neg_features)

# print("Positive Predictions: " + str(pos_result))
true_positives = cv2.countNonZero(pos_result)
false_negatives = len(pos_result) - true_positives

# print("Negative Predictions: " + str(neg_result))
false_positives = cv2.countNonZero(neg_result)
true_negatives = len(neg_result) - false_positives

print("True Positives: " + str(true_positives), "False Positives: " + str(false_positives))
print("True Negatives: " + str(true_negatives), "False Negatives: " + str(false_negatives))

precision_pos = float(true_positives) / (true_positives + false_positives)
precision_neg = float(true_negatives) / (true_negatives + false_negatives)
precision = (precision_pos + precision_neg) / 2

print("Precision: " + str(precision))
