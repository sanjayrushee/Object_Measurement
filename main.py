import cv2
from tkinter import *


root = Tk()

root.geometry("1360x710")
root.title("Object Measurement")
def objectMeasurement():
    button_image = PhotoImage(file="btn2.png")
    button = Button( image=button_image , bd=0, highlightthickness=0, bg="white", activebackground="#ffffff")
    button.place(x=482, y=600)


    cap = cv2.VideoCapture(0)

    pixel_to_cm = 0.05  

    min_area_threshold = 100  

    while True:
        
        ret, frame = cap.read()
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect edges using Canny edge detection
        edges = cv2.Canny(blurred, 50, 150) 

        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            contour_area = cv2.contourArea(contour)
            
            if contour_area > min_area_threshold:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                length_cm = w * pixel_to_cm
                breadth_cm = h * pixel_to_cm
                
                cv2.putText(frame, f"Length: {length_cm:.2f} cm", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                cv2.putText(frame, f"Breadth: {breadth_cm:.2f} cm", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Result", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    





bg = PhotoImage( file = "background.png") 
  
# Show image using label 
label1 = Label( root, image = bg) 
label1.place(x = 0,y = 0)
button_image = PhotoImage(file="btn.png")
button = Button( image=button_image,command = objectMeasurement , bd=0, highlightthickness=0, bg="#6325CE", activebackground="#6325CE")
button.place(x=482, y=600)

root.mainloop()