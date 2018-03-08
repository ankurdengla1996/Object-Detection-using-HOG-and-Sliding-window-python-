from sklearn import svm
import numpy as np
from pathlib import Path
import cv2
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree 

clf = svm.LinearSVR()
clf2 = KNeighborsClassifier() 
clf3 = tree.DecisionTreeClassifier(criterion = "gini",splitter = "best")
#clf4 = GaussianMixture()

x = []
y = []

classes = ['Bow','Cup','Gun','Spray','Telephone','Torch']

for j in range (7):
    pathlist = Path('./HOG/HOGFeatures'+str(j)+'/').glob('**/*.npy')

    for path in pathlist:
        path_in_str = str(path)
        h = np.load(path_in_str, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
        #print (h)
        x.append(h)
        if ( j == 3 ):
            y.append(1)
        else:
            y.append(-1)
        #print(j)
        #y.append(classes[j])

print(y)
clf.fit(x,y)
#clf2.fit(x,y)
clf3.fit(x,y)
#clf4.fit(x,y)

print(clf.predict(x))
#print(clf2.predict(x))
#print(clf3.predict(x))
#print(clf4.predict(x))

joblib.dump(clf, 'svm_model.sav')
#joblib.dump(clf2, 'knn_model.sav')
#joblib.dump(clf3, 'dtree_model.sav')