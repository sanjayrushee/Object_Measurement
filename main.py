import cv2
import utlis

# Set up webcam
webcam = True

# Open the webcam
cap = cv2.VideoCapture(0)

# Set the resolution of the capture
cap.set(3, 1920)
cap.set(4, 1080)

# Define the scale for measurements
scale = 3

while True:
    # Read the frame from the webcam
    success, img = cap.read()

    if not success:
        print("Failed to read from webcam")
        break

    # Get contours from the image
    img, finalContours = utlis.getContours(img, minArea=1000, filter=4, draw=False)

    # Measure objects
    for cnt in finalContours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # Filter out small contours
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f'Width: {w / scale:.2f} cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img, f'Height: {h / scale:.2f} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the original image with contours and measurements
    cv2.imshow('Camera View', img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture device
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
