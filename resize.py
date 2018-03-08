import cv2
import os
from pathlib import Path
import numpy as np
from PIL import Image

for j in range (7):    
    pathlist = Path('./cropdata/'+str(j)+'/').glob('**/*.jpg')
    pathresize = './Resized/' + str(j) + '/' 

    if not os.path.exists(pathresize):
            os.makedirs(pathresize)

    i = 1
    for path in pathlist:
        path_in_str = str(path)
        im = Image.open(path_in_str)
        imResize = im.resize((40,40), Image.ANTIALIAS)

        imResize.save( pathresize + str(i) + '.jpg', 'JPEG', quality=90)
        i = i + 1