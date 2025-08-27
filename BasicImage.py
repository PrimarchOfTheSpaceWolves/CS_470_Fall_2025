import numpy as np
import cv2

def main():
    image = np.zeros((480,640,3), dtype="uint8")
    
    image[100:250,300:500,2] = 255
    
    other_image = np.copy(image[50:250,300:500,:])
    other_image[:,:,:] = 255
    
    
    cv2.imshow("IMAGE", image)
    cv2.imshow("OTHER", other_image)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    