import cv2
import numpy as np 
import imutils
import math
from imutils.video import VideoStream
from imutils import face_utils
from skimage import measure
import time

print("[INFO] opening camera stream")
vs = VideoStream().start() 
time.sleep(2.0)

while True:
    #Grab the frame
    frame = vs.read()
    
    #Resize the image and conver to grayscale
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Apply Gaussian Blur 
    #Later obtain Gaussian Kernel from:
    #cvs.getGaussianKernel()
    blur = cv2.GaussianBlur(gray ,(11,11),0)
    
    #Apply threshold to get light intensity
    th = cv2.threshold(blur, 200,255, cv2.THRESH_BINARY)[1]
    #cv2.imshow("Threshold",th)
    #cv2.waitkey(0)
    
    """
    Cleaning up the frame with erosions and dilations
    
    th = cv2.erode(th,None,iterations=2)
    th = cv2.dilate(th,None,iterations=4)
    """
    
    labels = measure.label(th, neighbors=8, background=0)
    mask = np.zeros(th.shape, dtype="uint8")
     
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(th.shape, dtype="uint8")
        labelMask[labels == label] = 255
        
        numPixels = cv2.countNonZero(labelMask)
        
        if numPixels > 500:
            mask = cv2.add(mask, labelMask)
                
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)
        #((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 1)
        cv2.putText(frame, "Phone??", (x+6,y-6),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
                             
    # show the output image
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

