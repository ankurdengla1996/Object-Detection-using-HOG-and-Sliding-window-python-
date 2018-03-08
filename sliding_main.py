import multiprocessing as mp
from helpers2 import sliding_window
import argparse
import cv2
from sklearn.externals import joblib
from PIL import Image

def sl_window (ydata):

    global image
    filepath = ['./cup_svm_model.sav','./cup_svm_model.sav','./cup_svm_model.sav','./cup_svm_model.sav']
    loaded_model =  joblib.load(filepath[ydata[0]]) 
    hog = cv2.HOGDescriptor('a.xml')
    (winW, winH) = (40, 40)
    
    xmax = 0
    ymax = 0
    ma = -2

    ftest = []
    n = ydata[0]

    for (x, y, window) in sliding_window(image=image, ystart=ydata[1], yend=ydata[2],  stepSize=5, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        im = Image.fromarray(window)
        im.save("w"+str(n)+".jpg")

        img = Image.open('w'+str(n)+'.jpg')
        imResize = img.resize((40,40), Image.ANTIALIAS)		
        imResize.save( 'w2'+str(n)+'.jpg', 'JPEG', quality=90)

        im2 = cv2.imread('w2'+str(n)+'.jpg')
        h = hog.compute(im2)
        h2 = [x for sublist in h for x in sublist]

        ftest.append(h2)

        ans = loaded_model.predict(ftest)
        if  ans > ma :
            ma = ans
            (xmax,ymax) = (x,y)
        ftest.pop()
    
    return (xmax,ymax,ma)

def sliding_window_0():
    global args
    global image
    filepath = './cup_svm_model.sav'
    loaded_model =  joblib.load(filepath) 
    hog = cv2.HOGDescriptor('a.xml') 

    image = cv2.imread(args["image"])
    (winW, winH) = (40, 40)
    y = [(0,0,int(image.shape[0]/4)),(1,int(image.shape[0]/4)-35,int(2*image.shape[0]/4)-35),(2,int(2*image.shape[0]/4)-35,int(3*image.shape[0]/4)),(3,int(3*image.shape[0]/4)-35,image.shape[0])]
    #y = [(0,0,int(image.shape[0]/2)),(1,int(image.shape[0]/2)-35,image.shape[0])]
    #y = [(0,0,image.shape[0])]

    with mp.Pool(4) as p:
        ans = (p.map(sl_window,y ))

    ma = 0

    for ( a, b, arr ) in ans:
        if ( arr[0] > ma):
            xmax = a
            ymax = b

    print ( xmax, ymax )

    clone = image.copy()
    cv2.rectangle(image, (xmax, ymax), (xmax+45, ymax+45), (0, 0, 0), thickness=2)
    cv2.imshow("Detections", image)
        
    return (xmax, ymax)

def sliding_window_1():
    global args
    global image
    filepath = './cup_svm_model.sav'
    loaded_model =  joblib.load(filepath) 
    hog = cv2.HOGDescriptor('a.xml') 

    image = cv2.imread(args["image"])
    (winW, winH) = (40, 40)
    y = [(0,0,int(image.shape[0]/4)),(1,int(image.shape[0]/4)-35,int(2*image.shape[0]/4)-35),(2,int(2*image.shape[0]/4)-35,int(3*image.shape[0]/4)),(3,int(3*image.shape[0]/4)-35,image.shape[0])]
    #y = [(0,0,int(image.shape[0]/2)),(1,int(image.shape[0]/2)-35,image.shape[0])]
    #y = [(0,0,image.shape[0])]

    with mp.Pool(4) as p:
        ans = (p.map(sl_window,y ))

    ma = 0

    for ( a, b, arr ) in ans:
        if ( arr[0] > ma):
            xmax = a
            ymax = b

    print ( xmax, ymax )

    clone = image.copy()
    cv2.rectangle(image, (xmax, ymax), (xmax+45, ymax+45), (0, 0, 0), thickness=2)
    cv2.imshow("Detections", image)
        
    return (xmax, ymax)

def sliding_window_2():
    global args
    global image
    filepath = './cup_svm_model.sav'
    loaded_model =  joblib.load(filepath) 
    hog = cv2.HOGDescriptor('a.xml') 

    image = cv2.imread(args["image"])
    (winW, winH) = (40, 40)
    y = [(0,0,int(image.shape[0]/4)),(1,int(image.shape[0]/4)-35,int(2*image.shape[0]/4)-35),(2,int(2*image.shape[0]/4)-35,int(3*image.shape[0]/4)),(3,int(3*image.shape[0]/4)-35,image.shape[0])]
    #y = [(0,0,int(image.shape[0]/2)),(1,int(image.shape[0]/2)-35,image.shape[0])]
    #y = [(0,0,image.shape[0])]

    with mp.Pool(4) as p:
        ans = (p.map(sl_window,y ))

    ma = 0

    for ( a, b, arr ) in ans:
        if ( arr[0] > ma):
            xmax = a
            ymax = b

    print ( xmax, ymax )

    clone = image.copy()
    cv2.rectangle(image, (xmax, ymax), (xmax+45, ymax+45), (0, 0, 0), thickness=2)
    cv2.imshow("Detections", image)
        
    return (xmax, ymax)

def sliding_window_3():
    global args
    global image
    filepath = './cup_svm_model.sav'
    loaded_model =  joblib.load(filepath) 
    hog = cv2.HOGDescriptor('a.xml') 

    image = cv2.imread(args["image"])
    (winW, winH) = (40, 40)
    y = [(0,0,int(image.shape[0]/4)),(1,int(image.shape[0]/4)-35,int(2*image.shape[0]/4)-35),(2,int(2*image.shape[0]/4)-35,int(3*image.shape[0]/4)),(3,int(3*image.shape[0]/4)-35,image.shape[0])]
    #y = [(0,0,int(image.shape[0]/2)),(1,int(image.shape[0]/2)-35,image.shape[0])]
    #y = [(0,0,image.shape[0])]

    with mp.Pool(4) as p:
        ans = (p.map(sl_window,y ))

    ma = 0

    for ( a, b, arr ) in ans:
        if ( arr[0] > ma):
            xmax = a
            ymax = b

    print ( xmax, ymax )

    clone = image.copy()
    cv2.rectangle(image, (xmax, ymax), (xmax+45, ymax+45), (0, 0, 0), thickness=2)
    cv2.imshow("Detections", image)
        
    return (xmax, ymax)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

pt = []
pt.append(sliding_window_0())
pt.append(sliding_window_1())
pt.append(sliding_window_2())
pt.append(sliding_window_3())

print (pt)