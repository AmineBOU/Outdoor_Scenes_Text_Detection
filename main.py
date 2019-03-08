import cv2
import numpy as np
import os, re

from scipy.spatial.distance import dice
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score

# Evaluation method
def eval_results(expected_mask, estimated_mask):

	im1 = np.asarray(expected_mask).astype(np.bool)
	im2 = np.asarray(estimated_mask).astype(np.bool)
	if im1.shape != im2.shape:
		raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

	# Compute Dice coefficient
	intersection = np.logical_and(im1, im2)
	dice_score = 2. * intersection.sum() / (im1.sum() + im2.sum())
	jacc_coef  = jaccard_similarity_score(expected_mask, estimated_mask)

	return dice_score, jacc_coef


# list test files
data_path = 'data/test/'
files = [f for f in os.listdir(data_path) if re.match(r'.*\.JPG', f)]

results_path = 'results/test/'

# Generate expected masks

for file in files :
	img_num = file[:-4]
	try :
		os.system("python display_img_mask.py --input " + data_path + file)
	except:
		print(img_num)


# Estimate SWT & EAST mask
files = [f for f in os.listdir(results_path) if re.match(r'.*\.png', f)]

for file in files:

	# Retieve img id
	img_num = file[:-9]

	# 
	try :
		os.system("python swt/text_detect.py --image " + data_path + img_num + '.JPG')
		os.system("python east/main.py --input " + data_path + img_num + '.JPG ' + "--model east/frozen_east_text_detection.pb")
	except:
		print(img_num)
	pass

swt_files      = [f for f in os.listdir('./results/swt/') if re.match(r'.*\.png', f)]
east_files     = [f for f in os.listdir('./results/east/') if re.match(r'.*\.png', f)]
expected_files = [f for f in os.listdir('./results/test/') if re.match(r'.*\.png', f)]

swt_dice_list, swt_jacc_list = [], []
east_dice_list, east_jacc_list = [], []

for file in swt_files:
	try :
		file_num = file[:-13]

		swt_mask_path = './results/swt/' + file_num + '_swt_mask.png'
		east_mask_path = './results/east/' + file_num + '_east_mask.png'
		expected_mask_path = './results/test/' + file_num + '_mask.png'

		
		expected_mask  = cv2.imread(expected_mask_path, 0)
		swt_estimated_mask = cv2.imread(swt_mask_path, 0)
		east_estimated_mask = cv2.imread(east_mask_path, 0)

		dice_1, jacc_1 = eval_results(expected_mask, swt_estimated_mask)
		swt_dice_list.append(dice_1)
		swt_jacc_list.append(jacc_1)

		dice_2, jacc_2 = eval_results(expected_mask, east_estimated_mask)
		east_dice_list.append(dice_2)
		east_jacc_list.append(jacc_2)

	except:
		print(file_num)

print('SWT \t DICE', np.mean(swt_dice_list), '\t JACC ', np.mean(swt_jacc_list))
print('EAST \t DICE', np.mean(east_dice_list), '\t JACC ', np.mean(east_jacc_list))
