from helpers import pyramid
from helpers import sliding_window
import argparse
import time
import math
import cv2
from sklearn.externals import joblib
from PIL import Image
from pathlib import Path

'''ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
'''
filepath = './torch_svm_model2.sav'
loaded_model =  joblib.load(filepath) 
hog = cv2.HOGDescriptor('a.xml') 

pathlist = Path('./torchimages/').glob('**/*.jpg')
k = 0

cups = [(196, 178), (159, 165), (213, 175), (220, 190), (142, 167), (166, 178), (138, 164), (162, 165), (161, 159), (141, 171), (156, 171), (152, 187), (169, 178), (168, 176), (199, 171), (137, 167), (155, 126), (163, 168)]
telephones = [(256, 182), (248, 161), (247, 161), (221, 178), (173, 171), (272, 178), (265, 197), (203, 187), (252, 182), (245, 172), (274, 177), (202, 173), (264, 183), (245, 171), (203, 175), (222, 179)]
#sprays = [(140, 182), (138, 167), (,), (,), (,), (,), (,), (,), (,), (,), (,), (,), (,), (,), (,), (,)]
torches = [(140, 173), (138, 167), (174, 161), (132, 162), (131, 159), (235, 166), (141, 169), (145, 163), (140, 169), (131, 161), (184, 176), (132, 161), (128, 173), (142, 173)]

i = 0

avg = 0

for path in sorted(pathlist):
	path_in_str = str(path)
	#path_in_str = './out3.jpg'
	image1 = cv2.imread(path_in_str)
	image = cv2.resize(image1,(320,240))
	# cv2.imwrite('abc.jpg',image)
	(winW, winH) = (40, 40)

	xmax = -1
	ymax = -1
	ma = -2
	ftest = []
	detections = []
	#p = 0

	#for resized in pyramid(image, scale=1.5):
	for (x, y, window) in sliding_window(image, stepSize=35, windowSize=(winW, winH)):
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		#im = window
		#imResize = im.resize((150,150), Image.ANTIALIAS)
		
		h = hog.compute(window)
		h2 = [x for sublist in h for x in sublist]
		#print(len(h2))
		ftest.append(h2)

		ans = loaded_model.predict(ftest)
		if  ans > ma :
			ma = ans
			(xmax,ymax) = (x,y)
		#print(ans)
		ftest.pop()

		if ( ans > 0 ):
			cv2.imwrite('./Detections/'+'xyz'+str(k)+'.jpg',window)
			k+=1
			detections.append((x,y)) 

		clone = image.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)


	x1 = torches[i][0]
	y1 = torches[i][1]

	dist = math.hypot(x1 - xmax, y1 - ymax)
	dist /= 4

	print (i, (xmax,ymax), (torches[i][0],torches[i][1]), dist)

	avg += dist

	i += 1

	# for (x,y) in detections:
	#  	cv2.rectangle(image, (x, y), (x+40, y+40), (0, 0, 0), thickness=2)
	
	# cv2.rectangle(image,(xmax,ymax), (xmax+40,ymax+40), (0, 0, 0), thickness=2)
	# cv2.imshow("Detections", image)
	# cv2.waitKey()
	# cv2.destroyWindow("Detections")

avg /= 18
print ( avg )