import cv2 
import numpy as np
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import math 
import time

def initializeBlobDetector():
    
    """
    Creating a blob detector 
    """
    print("[INFO} Initializing blobby the unicorn")

    params = cv2.SimpleBlobDetector_Params()
    #Threshold Values set min, max and step
    #Reconsider setting these values, they might not be required
    params.minThreshold = 200;
    params.maxThreshold = 5000;
    
    #Color filter set color in 0,255 range
    #params.filterByColor = True
    #params.blobColor = 0
    
    #Area Filter set min and max
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 750
    
    #Cricularity Filer set min and max 
    params.filterByCircularity = True
    params.minCircularity = .65
    params.maxCircularity = .85

    #Convexity filter set min
    params.filterByConvexity = False
    params.minConvexity = 0 

    #Inertia filter set max and min
    params.filterByInertia = True
    params.minInertiaRatio = .1
    params.maxInertiaRatio = .7

    #Create the detector instance
    detector = cv2.SimpleBlobDetector_create(params)
    
    return detector

def main():
    
    print("[INFO] rendering a world of pure imagination")

    #Grab the frame
    img = cv2.imread('data/distract.png')

    #Crop out the ROI
    #cv2.imshow("Image",img)
    x = 250;y = 120;h = 250
    img = img[y:y+h, x:x+h]
    #cv2.imshow("Cropped",img)

    #Image downize by half
    img_s = cv2.resize(img, (0,0), fx=0.75, fy=0.75)
    img_g = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
    #Gaussian Blurring
    blur = cv2.GaussianBlur(img_g,(7,7),0)
    #cv2.imshow("Blurred",blur)
    
    #LAB format split, CLAHE Histogram Equalization and merge->2 BGR
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    clh = cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    #cv2.imshow("Clahe",clh)
    
    #Convert to hsv and perform thresholding of hsv vals
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    low = np.array([0,0,0])
    high = np.array([180,210,210])

    mask = cv2.inRange(hsv,low,high)
    #res = cv2.bitwise_and(hsv,hsv,mask=mask)
    cv2.imshow("mask",mask)
    #cv2.imshow("Filtered",res)
    
    #Apply adaptive threshold
    ret, mask = cv2.threshold(blur,210,255,cv2.THRESH_BINARY)
    

    """
    #Color histogrammization on obtained hsv mask filter
    hist = cv2.calcHist([hsv],[0],mask,[180],[0,180])
    cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
    hist = cv2.calcBackProject([hsv],[0],hist,[0,180],1)
    cv2.imshow("Post Hist",hist)
    """

    #Apply blobby the unicorn
    detector = initializeBlobDetector()
    phone_points = detector.detect(mask)
    # draw the bright spot on the image
    for p in phone_points : 
        x=int(p.pt[0]);y=int(p.pt[1]);h=int(p.size)
        cv2.rectangle(img_s, (x-h,y-h), (x+h,y+h), (0, 0, 255), 1)
        cv2.putText(img_s, "Phone??", (x-h,y-h),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)

    cv2.imshow("Detect",img_s)
    key = cv2.waitKey(0)
    if(key == ord("q") or key == ord("c")):
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()
