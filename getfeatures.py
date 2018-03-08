import cv2
from pathlib import Path
import numpy as np
from PIL import Image
import os

hog = cv2.HOGDescriptor('a.xml')

for j in range (7):
    pathfile = './HOG/HOGFeatures' + str(j) + '/HOGimage_'

    hogpath = './HOG/HOGFeatures' + str(j) + '/'

    if not os.path.exists(hogpath):
        os.makedirs(hogpath)

    i = 1
    #pathlist = Path('./Resize'+str(j)+'/').glob('**/*.jpg')
    pathlist = Path('./Resized/'+str(j)+'/').glob('**/*.jpg')

    for path in pathlist:
        print(path)
        path_in_str = str(path)
        im = cv2.imread(path_in_str)
        h = hog.compute(im)
        #hog.save('a.xml')
        #print (len(h))
        h2 = [x for sublist in h for x in sublist]

        file = pathfile + str(i)
        i = i + 1
        np.save(file,h2,allow_pickle=True,fix_imports=True)