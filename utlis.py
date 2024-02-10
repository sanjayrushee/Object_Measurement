import cv2
import numpy as np

def getContours(img, minArea=1000, filter=0, draw=False):
    # Check if the input image is empty
    if img is None:
        print("Error: Input image is empty")
        return None, []

    # Check if the input image is grayscale
    if len(img.shape) == 2:
        imgGray = img
    else:
        # Convert image to grayscale
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    
    # Perform Canny edge detection
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    
    # Dilate the edges
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and number of vertices
    finalContours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == filter:
                finalContours.append(cnt)
                if draw:
                    cv2.drawContours(img, cnt, -1, (255, 0, 255), 7)

    return img, finalContours
