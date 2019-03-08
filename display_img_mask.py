import os
import cv2, skimage
import numpy as np
import matplotlib.pyplot as plt
import argparse
import ntpath

# Read img and correspoding boxs
def read_img_boxs(img_path):
  
    file_name = os.path.splitext(img_path)[0]

    img = cv2.imread(img_path)

    boxs = open(file_name + '.gt').read().split('\n')
    boxs.pop()

    return img, boxs


# Compute points of each box using rotation transformation
def compte_box_points(x, y, w, h, theta):
    centre = (x + w/2, y + h/2)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    p1 = [ + w / 2,  + h / 2]
    p2 = [- w / 2,  + h / 2]
    p3 = [ - w / 2, - h / 2]
    p4 = [ + w / 2,  - h / 2]

    p1_new = np.dot(p1, R) + centre
    p2_new = np.dot(p2, R) + centre
    p3_new = np.dot(p3, R) + centre
    p4_new = np.dot(p4, R) + centre

    points = np.array([p1_new, p2_new, p3_new, p4_new])

    return points

# Convert box to an array
def boxs_to_array(boxs):
    stored_pts = []
    for box in boxs :
        x = float(box.split()[2])
        y = float(box.split()[3])
        w = float(box.split()[4])
        h = float(box.split()[5])
        theta = float(boxs[0].split()[6])


        points = compte_box_points(x, y, w, h, -theta)
        stored_pts.append(points)

        rect = cv2.polylines(img, np.int32([points]), 1, (0, 255, 0))

    return rect, stored_pts
  
# Generate mask
def generate_mask(img, stored_pts):
    mask = np.zeros(img.shape[:2])
   
    for i in range(len(stored_pts)):
        points = stored_pts[i]
        mask = cv2.drawContours(mask, np.int32([points]), 0, 255, thickness = -1)
    return mask


#------------------ Main ------------------#

# Parse input image
parser = argparse.ArgumentParser(description='Use this script to run TensorFlow implementation (https://github.com/argman/EAST) of EAST: An Efficient and Accurate Scene Text Detector (https://arxiv.org/abs/1704.03155v2)')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

args = parser.parse_args()
img_path  = args.input

img, boxs = read_img_boxs(img_path)
rect, stored_pts = boxs_to_array(boxs)
cv2.imwrite('img_with_boxs.png', rect)
cv2.imshow('image', rect)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Expected mask
mask = generate_mask(img, stored_pts)
cv2.imwrite('results/test/' + ntpath.basename(img_path)[:-4]+'_mask.png', mask)
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()