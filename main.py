import cv2
import numpy as np

def getContours(img, minArea=1000, filter=0, draw=False):
    # Check if the input image is empty
    if img is None:
        print("Error: Input image is empty")
        return None, []

    # Convert image to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
    
    # Perform adaptive thresholding
    _, imgThresh = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Perform morphological operations
    kernel = np.ones((5, 5), np.uint8)
    imgMorph = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(imgMorph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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
                    cv2.drawContours(img, [cnt], -1, (255, 0, 255), 2)

    return img, finalContours
