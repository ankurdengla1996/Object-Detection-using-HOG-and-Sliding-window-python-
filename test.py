import cv2
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.externals import joblib

# Resize
pathlist = Path('./Test/Images/').glob('**/*.jpg')
pathresize = './Test/Resize/'

i = 1
for path in pathlist:
    path_in_str = str(path)
    im = Image.open(path_in_str)
    imResize = im.resize((150,150), Image.ANTIALIAS)
    imResize.save( pathresize + str(i) + '.jpg', 'JPEG', quality=90)
    i = i + 1

# HOG

hog = cv2.HOGDescriptor('a.xml')
ftest = []
pathfile = './Test/HOG/HOGimage_'
pathlist = Path('./Test/Resize/').glob('**/*.jpg')

for path in pathlist:
    #path_in_str = str(path)
    path_in_str = './out3.jpg'
    im = cv2.imread(path_in_str)
    h = hog.compute(im)
    print(path_in_str)
    h2 = [x for sublist in h for x in sublist]
    ftest.append(h2)

# Test
#filepath = './svm_model.sav'
#loaded_model =  joblib.load(filepath)
#loaded_model2 =  joblib.load('./knn_model.sav')
loaded_model3 =  joblib.load('./dtree_model.sav')

#print(loaded_model.predict(ftest))
#print(loaded_model2.predict(ftest))
print(loaded_model3.predict(ftest))