import cv2
from sklearn.externals import joblib
from pathlib import Path
import sklearn

hog = cv2.HOGDescriptor('a.xml') 

loaded = joblib.load('torch_svm_model2.sav')
pathlist = Path('./Testing Data/4/').glob('**/*.jpg')

x = []

for path in sorted(pathlist):
    path_in_str = str(path)
    image = cv2.imread(path_in_str)

    h = hog.compute(image)
    h2 = [a for sublist in h for a in sublist]

    x.append(h2)
    print(path_in_str)

y2 = loaded.predict(x)
y3 = []

for a in y2:
    if ( a > -0.1 ):
        y3.append(1)
    else:
        y3.append(-1)

#y = [1,1,1,-1,1,1,1,1,1,1,-1,1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
#y = [1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
#y = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,] 
y = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,]

print(y,y3,sklearn.metrics.f1_score(y,y3),sklearn.metrics.mean_squared_error(y,y2),sep='\n')

# cup -> rmse = 0.4088  fscore = 0.7894 cup_model
# telephone -> rmse = 0.2986 fscore = 0.902 telephone_model
# spray -> rmse = 0.4689 fscore = 0.7586 spray_model    
# torch -> rmse = 0.3051 fscore = 0.9285 torch_model2