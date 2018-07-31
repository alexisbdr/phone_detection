import cv2 
import numpy as np
import math 
import time

tracked_points = []

def calc_dist(tp,newp):
    """
    Calculate 3D euclidean distance 
    """
    return np.linalg.norm(tp-newp, axis = 1)

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
    params.minArea = 60
    params.maxArea = 700
    
    #Cricularity Filer set min and max 
    params.filterByCircularity = True
    params.minCircularity = .6
    params.maxCircularity = .85

    #Convexity filter set min
    params.filterByConvexity = False
    params.minConvexity = 0 

    #Inertia filter set max and min
    params.filterByInertia = True
    params.minInertiaRatio = .5
    params.maxInertiaRatio = 1

    #Create the detector instance
    detector = cv2.SimpleBlobDetector_create(params)
    
    return detector

def main():
    
    print("[INFO] rendering a world of pure imagination")
    
    tp=np.array([0,0,0])
    count=0
    max_count = 25
    detector = initializeBlobDetector()
    cap = cv2.VideoCapture('data/glare.mp4')
    
    if( not cap.isOpened()):
        print("[ALERT] COULD NOT OPEN VIDEO FILE")
    
    while(cap.isOpened()):
        
        #Grab the frame
        ret, frame = cap.read()
        
        if ret:
            count+=1
            
            #ROI cropping
            #Crop #1 to extract video frame from template image
            x=290;y=70;h=350
            frame = frame[y:y+h, x:x+h]
            cv2.imshow("Frame",frame)
            
            #Crop #2 extract ROI from frame
            x = 160;y = 70;h = 170
            img = frame[y:y+h, x:x+h]
            cv2.imshow("Cropped",img)

            #Image downize by half
            img_s = cv2.resize(img, (0,0), fx=0.75, fy=0.75)
            img_g = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
            
            #Gaussian Blurring
            blur = cv2.GaussianBlur(img_g,(5,5),0)
            #blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR) 
            
            """
            #Convert to LAB format
            lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(lab)
            #Apply Clahe Histogram equalization
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            for i in range(len(lab_planes)):
                lab_planes[i] = clahe.apply(lab_planes[i])
            #Convert back to bgr
            lab = cv2.merge(lab_planes)
            clh = cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
            cv2.imshow("Clahe",clh)
            """
            #Apply CLAHE
            #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            #clh = clahe.apply(blur)
            
            """
            HSV FILTERING
            
            #Convert to hsv and perform thresholding of hsv vals
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            low = np.array([0,0,0])
            high = np.array([180,210,210])
            mask = cv2.inRange(hsv,low,high)
            #res = cv2.bitwise_and(hsv,hsv,mask=mask)
            cv2.imshow("mask",mask)
            #cv2.imshow("Filtered",res)
            """
            
            #Apply adaptive threshold
            ret, mask = cv2.threshold(blur,215,255,cv2.THRESH_BINARY)
            cv2.imshow("Mask", mask)
            mask = 255-mask 
            """
            #Color histogrammization on obtained hsv mask filter
            hist = cv2.calcHist([hsv],[0],mask,[180],[0,180])
            cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
            hist = cv2.calcBackProject([hsv],[0],hist,[0,180],1)
            cv2.imshow("Post Hist",hist)
            """
            
            #Apply blobby the unicorn
            phone_points = detector.detect(mask)
            
            # draw the bright spot on the image
            for p in phone_points : 
                x=p.pt[0];y=p.pt[1];h=p.size;newp=np.array([x,y,h])
                tp = np.vstack([tp,newp])
                print(calc_dist(tp[-max_count:],newp))
                #for tp in tracked_points : 
                    #print(calc_dist)
                x=int(x);y=int(y);h=int(h)
                cv2.rectangle(img_s, (x-h,y-h), (x+h,y+h), (0, 0, 255), 1)
                cv2.putText(img_s, "Phone??", (x-h,y-h),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
        
            cv2.imshow("Detect",img_s)
            key = cv2.waitKey(0)
            if(key == 32):
                continue
            elif(key==ord("q") or key==ord("c") or key == 27): 
                cv2.destroyAllWindows()
                break

    cap.release()
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()
