import cv2
import os
from pathlib import Path
import numpy as np

pathlist = Path('./torch/').glob('**/*.jpg')
pathresize = './torch2/' 

i = 1
for path in pathlist:
    path_in_str = str(path)
    im = cv2.imread(path_in_str)
    imResize = cv2.resize(im,(320,240))

    path2 = pathresize + str(i) + '.jpg'

    cv2.imwrite(path2,imResize)
    i = i + 1