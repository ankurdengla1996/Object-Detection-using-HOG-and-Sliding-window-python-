from ffmpy import FFmpeg
from pathlib import Path
import cv2
import os

pathlist = Path('./data4/').glob('**/*.avi')
j = 0

for path in pathlist:
    path_in_str = str(path)
    ff = FFmpeg(inputs={path_in_str:None},outputs={'./torch/out'+str(j)+'_%d.jpg':['-vf','fps=4']})
    j = j+1
    print(ff.cmd)
    ff.run()
