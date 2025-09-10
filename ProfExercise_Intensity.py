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
from torch import nn 
from torchvision.transforms import v2

class IntTransform(Enum):
    ORIGINAL = 0
    NEGATIVE = 1
    HISTEQUAL = 2

def transform(image, chosenT):
    
    if chosenT == IntTransform.ORIGINAL:    
        output = np.copy(image)
    elif chosenT == IntTransform.NEGATIVE:
        output = 255 - image
    elif chosenT == IntTransform.HISTEQUAL:
        output = cv2.equalizeHist(image)
    
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
    
    chosen_T = 0
    #for item in list(IntTransform):
    #    print("*", item.name, "=", item.value)
    #chosenT = IntTransform(int(input("Enter choice: ")))
    
    conv_layer = nn.Conv2d(in_channels=3, out_channels=1,
                           kernel_size=1, bias=False)
    model = nn.Sequential(conv_layer)
    print(model)
    
    device = "cuda" # "mps" # "cpu"
    model = model.to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    data_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])    
        
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
        key = -1
        while key == -1:
            # Get next frame from camera
            _, image = camera.read()
                        
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            #grayscale = cv2.equalizeHist(grayscale)
            
            
            grayscale = np.expand_dims(grayscale, axis=-1)
            desired_output = data_transform(grayscale)
            desired_output = torch.unsqueeze(desired_output, 0)
            #print(desired_output.shape)
            
            color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            data_input = data_transform(color)
            data_input = torch.unsqueeze(data_input, 0)
            
            model.train()
            data_input = data_input.to(device)
            desired_output = desired_output.to(device)
            
            pred_output = model(data_input)
            
            loss = loss_fn(pred_output, desired_output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            out_image = pred_output.detach().cpu().numpy()
            out_image = out_image[0]
            out_image = np.transpose(out_image, [1,2,0])
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
                                
            # Show the image
            cv2.imshow(windowName, grayscale)
            cv2.imshow("Output", out_image)
            
            print("Weights:", conv_layer.weight.detach().cpu().numpy())
            print("Loss:", loss.detach().cpu().numpy())
            
            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)

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
    
    # Not a real change

# The main function
if __name__ == "__main__": 
    main()
    