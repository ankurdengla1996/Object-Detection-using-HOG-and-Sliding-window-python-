from helpers import sliding_window
import cv2
from pathlib import Path
from PIL import Image

pathlist = Path('./ResizedData3/').glob('**/*.jpg')
j = 0

for path in pathlist:
    path_in_str = str(path)
    image = cv2.imread(path_in_str)

    for (x, y, window) in sliding_window(image, stepSize=40, windowSize=(40, 40)):
        if window.shape[0] != 40 or window.shape[1] != 40:
            continue

        im = Image.fromarray(window)
        im.save('./cropdata/6/'+str(j)+'.jpg')
        j += 1