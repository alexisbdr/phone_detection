import cv2 
import numpy as np 

cap = cv2.VideoCapture('data/glare.mp4')

if(not cap.isOpened()):
	print('[ALERT] COULD NOT OPEN VIDEO FILE')

while(cap.isOpened()):
        
        #Grab the frame
        ret, frame = cap.read()
        
        if ret: 
            #ROI cropping
            #Crop #1 to extract video frame from template image
            x=290;y=70;h=350
            frame = frame[y:y+h, x:x+h]
            cv2.imshow("Frame",frame)
            
            #Crop #2 extract ROI from frame
            x = 160;y = 70;h = 170
            img = frame[y:y+h, x:x+h]
            cv2.imshow("Cropped",img)

            key = cv2.waitKey(0)
            if(key == 32):
                continue
            elif(key==ord("q") or key==ord("c") or key == 27): 
                cv2.destroyAllWindows()
                break

cap.release()
cv2.destroyAllWindows()
            
