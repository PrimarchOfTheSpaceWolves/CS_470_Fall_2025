###############################################################################
# IMPORTS
###############################################################################

import sys
import numpy as np
import torch
import cv2
import pandas
import sklearn
from torch import nn

###############################################################################
# MAIN
###############################################################################

def main():   
    print("BEGINNING EXERCISE...")
         
    ###############################################################################
    # PYTORCH
    ###############################################################################
    
    b = torch.rand(5,3)
    print(b)
    print("Do you have Torch CUDA?:", torch.cuda.is_available())
    
    ###############################################################################
    # PRINT OUT VERSIONS
    ###############################################################################

    print("Torch:", torch.__version__)
    print("Numpy:", np.__version__)
    print("OpenCV:", cv2.__version__)
    print("Pandas:", pandas.__version__)
    print("Scikit-Learn:", sklearn.__version__)
    
    counter = 0
    MAX_COUNTER = 30
    lastImage = None
        
    ###############################################################################
    # OPENCV
    ###############################################################################
    if len(sys.argv) <= 1:
        # Webcam
        print("Opening the webcam...")

        # Linux/Mac (or native Windows) with direct webcam connection
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW recommended on Windows 
        # WSL: Use Yawcam to stream webcam on webserver
        # https://www.yawcam.com/download.php
        # Get local IP address and replace
        #IP_ADDRESS = "192.168.0.7"    
        #camera = cv2.VideoCapture("http://" + IP_ADDRESS + ":8081/video.mjpg")
        
        # Did we get it?
        if not camera.isOpened():
            print("ERROR: Cannot open the camera!")
            exit(1)

        # Create window ahead of time
        windowName = "Webcam"
        cv2.namedWindow(windowName)
        
        highT = 200
        lowT = 100        

        # While not closed...
        key = -1
        #while key == -1:
        while key != 27:
            # Get next frame from camera
            _, image = camera.read()
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(grayscale, lowT, highT)
            
            line_seg = cv2.HoughLinesP(canny, 1.0, np.pi/180.0, 100, 
                                       minLineLength=10.0,
                                       maxLineGap=10.0)
            
            edge_color = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
            
            for line in line_seg:
                cv2.line(edge_color, 
                         (line[0][0], line[0][1]),
                         (line[0][2], line[0][3]),
                         (0,0,255), 3)
               
            # Show the image
            cv2.imshow(windowName, grayscale)
            cv2.imshow("CANNY", canny)
            cv2.imshow("HOUGH", edge_color)
                        
            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)
            
            if(key == ord('a')): lowT -= 10
            if(key == ord('s')): lowT += 10
            if(key == ord('d')): highT -= 10
            if(key == ord('f')): highT += 10
            print(lowT, highT)

        # Release the camera and destroy the window
        camera.release()
        cv2.destroyAllWindows()

        # Close down...
        print("Closing application...")

    else:
        # Trying to load image from argument

        # Get filename
        filename = sys.argv[1]

        # Load image
        print("Loading image:", filename)
        image = cv2.imread(filename) 
        
        # Check if data is invalid
        if image is None:
            print("ERROR: Could not open or find the image!")
            exit(1)

        # Show our image (with the filename as the window title)
        windowTitle = "PYTHON: " + filename
        cv2.imshow(windowTitle, image)

        # Wait for a keystroke to close the window
        cv2.waitKey(-1)

        # Cleanup this window
        cv2.destroyAllWindows()
        
    print("PROGRAM COMPLETE.")

# The main function
if __name__ == "__main__": 
    main()
    