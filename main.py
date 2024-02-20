import cv2
import numpy as np

# Define the conversion factor from pixels to centimeters
# You need to measure this based on the real-world dimensions of the object
# For example, if 100 pixels = 5 cm, then the conversion factor is 5 / 100 = 0.05 cm/pixel
pixel_to_cm = 0.05

###################################
webcam = True
# path = '1.jpg'  # Removed line
cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 1920)
cap.set(4, 1080)
scale = 3
wP = 210 * scale
hP = 297 * scale

# Parameters for the moving average filter
N = 5  # Number of measurements to average
length_buffer = []  # Buffer for storing recent length measurements
breadth_buffer = []  # Buffer for storing recent breadth measurements
###################################

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 100)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDial, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

            # Calculate length and breadth in centimeters
            length_cm = w * pixel_to_cm
            breadth_cm = h * pixel_to_cm

            # Store recent measurements in the buffers
            length_buffer.append(length_cm)
            breadth_buffer.append(breadth_cm)

            # Keep only the last N measurements in the buffers
            if len(length_buffer) > N:
                length_buffer.pop(0)
            if len(breadth_buffer) > N:
                breadth_buffer.pop(0)

            # Calculate the average length and breadth
            avg_length = sum(length_buffer) / len(length_buffer)
            avg_breadth = sum(breadth_buffer) / len(breadth_buffer)

            # Display length and breadth on the image
            cv2.putText(img, f"Length: {avg_length:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(img, f"Breadth: {avg_breadth:.2f} cm", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow('Result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#this is testing line