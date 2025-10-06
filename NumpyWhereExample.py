import numpy as np

def main():
    image = np.array([
        [1,1,1],
        [1,3,2]
    ], dtype="uint8")

    print(image)
    
    coords = np.where(image == 1)
    print(coords)
    ycoords = coords[0]
    xcoords = coords[1]
    
    ymin = np.min(ycoords)
    ymax = np.max(ycoords)
    
    xmin = np.min(xcoords)
    xmax = np.max(xcoords)
    
    print(ymin, "to", ymax)
    print(xmin, "to", xmax)
    
    
    for i in range(len(ycoords)):
        print(ycoords[i], ",", xcoords[i])

if __name__ == "__main__":
    main()
    