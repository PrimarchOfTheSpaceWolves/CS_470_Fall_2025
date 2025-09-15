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
from torchvision.transforms import v2

class FilterType(Enum):
    BOX = 0
    GAUSS = 1
    MEDIAN = 2
    LAPLACE = 3
    SHARP = 4
    SOBEL_X = 5
    SOBEL_Y = 6
    GRAD_MAG = 7
    
def doFilter(image, fsize, ftype):
    
    if ftype == FilterType.BOX:
        output = cv2.boxFilter(image, -1, (fsize,fsize))
    elif ftype == FilterType.GAUSS:
        output = cv2.GaussianBlur(image, (fsize,fsize), sigmaX=0)
    elif ftype == FilterType.MEDIAN:
        output = cv2.medianBlur(image, fsize)
    elif ftype == FilterType.LAPLACE:
        output = cv2.Laplacian(image, cv2.CV_64F, None, fsize, 0.25)
        output = cv2.convertScaleAbs(output, alpha=0.5, beta=127.0)
    elif ftype == FilterType.SHARP:
        laplace = cv2.Laplacian(image, cv2.CV_64F, None, fsize, 0.25)
        fimage = image.astype("float64")
        fimage -= laplace
        output = cv2.convertScaleAbs(fimage)
    elif ftype == FilterType.SOBEL_X:
        sobelX = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, 
                           ksize=fsize, scale=0.25)
        output = cv2.convertScaleAbs(sobelX, alpha=0.5,
                                     beta=127.0)
    elif ftype == FilterType.SOBEL_Y:
        sobelY = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, 
                           ksize=fsize, scale=0.25)
        output = cv2.convertScaleAbs(sobelY, alpha=0.5,
                                     beta=127.0)
    elif ftype == FilterType.GRAD_MAG:
        sobelX = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, 
                           ksize=fsize, scale=0.25)
        sobelY = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, 
                           ksize=fsize, scale=0.25)
        gradImage = np.absolute(sobelX) + np.absolute(sobelY)
        output = cv2.convertScaleAbs(gradImage)
    
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
    
    conv_layer = nn.Conv2d(in_channels=1, out_channels=1,
                           kernel_size=3, bias=False,
                           padding="same")
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
    
    '''
    chosen_F = 0
    for item in list(FilterType):
        print("*", item.name, "=", item.value)
    chosen_F = FilterType(int(input("Enter choice: ")))
    chosen_size = int(input("Enter size: "))
    '''
            
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
                        
            #fimage = doFilter(grayscale, chosen_size, chosen_F)
            
            sobelX = cv2.Sobel(grayscale, cv2.CV_64F, dx=1, dy=0, ksize=3, scale=0.25)
        
            grayscale = np.expand_dims(grayscale, axis=-1)
            sobelX = np.expand_dims(sobelX, axis=-1)
            
            #print(grayscale.shape)
            #print(sobelX.shape)
            
            data_input = data_transform(grayscale)
            desired_output = data_transform(sobelX)
            
            data_input = torch.unsqueeze(data_input, 0)
            desired_output = torch.unsqueeze(desired_output, 0)             
        
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
            
            display_desired = desired_output.detach().cpu().numpy()
            display_desired = display_desired[0]
            display_desired = np.transpose(display_desired, [1,2,0])                                     
            
            print("Loss:", loss.detach().cpu().numpy())
                                        
            # Show the image
            #cv2.imshow(windowName, grayscale)
            #cv2.imshow("FILTERED", fimage)
            
            # Show the image
            cv2.imshow(windowName, grayscale)
            cv2.imshow("Output", out_image)
            cv2.imshow("Desired", display_desired)
                        
            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)
            
        print("Weights:", conv_layer.weight.detach().cpu().numpy())

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
        #windowTitle = "PYTHON: " + filename
        #cv2.imshow(windowTitle, image)
        
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelX = cv2.Sobel(grayscale, cv2.CV_64F, dx=1, dy=0, ksize=3, scale=0.25)
        
        grayscale = np.expand_dims(grayscale, axis=-1)
        sobelX = np.expand_dims(sobelX, axis=-1)
        
        print(grayscale.shape)
        print(sobelX.shape)
        
        data_input = data_transform(grayscale)
        desired_output = data_transform(sobelX)
        
        data_input = torch.unsqueeze(data_input, 0)
        desired_output = torch.unsqueeze(desired_output, 0)             
        
        epochCnt = 1000
        
        for epoch in range(epochCnt):            
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
            
            display_desired = desired_output.detach().cpu().numpy()
            display_desired = display_desired[0]
            display_desired = np.transpose(display_desired, [1,2,0])
                                            
            # Show the image
            cv2.imshow("Input", grayscale)
            cv2.imshow("Output", out_image)
            cv2.imshow("Desired", display_desired)
            cv2.waitKey(10)
            
            print("Epoch", (epoch+1), " - Loss:", loss.detach().cpu().numpy())

            
        print("Weights:", conv_layer.weight.detach().cpu().numpy())
        print("Loss:", loss.detach().cpu().numpy())

        # Wait for a keystroke to close the window
        cv2.waitKey(-1)

        # Cleanup this window
        cv2.destroyAllWindows()
        
    print("PROGRAM COMPLETE.")

# The main function
if __name__ == "__main__": 
    main()
    