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
from enum import Enum

class MorphType(Enum):
    ERODE = 0
    DILATE = 1
    OPEN = 2
    CLOSE = 3
    
def do_morph(image, morph_type, iterations):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(grayscale, 0, 255,
                                    cv2.THRESH_OTSU)
    # output = np.copy(thresh_image)
    struct_size = 3 #11 #3
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (struct_size,struct_size))
    
    if morph_type == MorphType.ERODE:
        output = cv2.erode(thresh_image, element, iterations=iterations)
    elif morph_type == MorphType.DILATE:
        output = cv2.dilate(thresh_image, element, iterations=iterations)
    elif morph_type == MorphType.OPEN:
        output = cv2.morphologyEx(thresh_image, 
                                  cv2.MORPH_OPEN,
                                  element,
                                  iterations=iterations)
    elif morph_type == MorphType.CLOSE:
        output = cv2.morphologyEx(thresh_image, 
                                  cv2.MORPH_CLOSE,
                                  element,
                                  iterations=iterations)
        
    return output    

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

        # While not closed...
        iterations = 1
        key = -1
        ESC_KEY = 27
        while key != ESC_KEY:
            # Get next frame from camera
            _, image = camera.read()
            
            output = do_morph(image, 
                              morph_type=MorphType.CLOSE,
                              iterations=iterations)
            

            # Show the image
            cv2.imshow(windowName, image)
            cv2.imshow("MORPH", output)
            
            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)
            
            if key == ord('a'): iterations -= 1
            if key == ord('d'): iterations += 1
            iterations = max(iterations, 0)

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
    