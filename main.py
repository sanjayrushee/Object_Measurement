import cv2
from tkinter import *
import threading 
from PIL import Image, ImageTk

# Global variable to control the thread
running = False

def objectMeasurement(label):
    global running
    cap = cv2.VideoCapture(0)
    pixel_to_cm = 0.05  
    min_area_threshold = 100  

    while running:
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

        # Convert the frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to ImageTk format
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)

        # Update the label with the new frame
        label.config(image=frame)
        label.image = frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
def btn_clicked():
    global running 
    if not running:
        button.config(image=cancel)
        running = True
        video_label.config(width=900, height=430)
        threading.Thread(target=objectMeasurement, args=(video_label,)).start()
    else:
        button.config(image=measure_now)
        running = False

root = Tk()
root.geometry("1360x710")
root.title("Object Measurement")

# Load background image
bg_image = PhotoImage(file="background.png")
background_label = Label(root, image=bg_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create a label to display the video feed
video_label = Label(root)
video_label.place(x=220, y=95)  # Adjust these coordinates as needed

measure_now = PhotoImage(file="measure.png")
cancel = PhotoImage(file="cancel.png")
running = False
button = Button(image=measure_now, command=btn_clicked, bd=0, highlightthickness=0, bg="#6325CE", activebackground="#6325CE")
button.place(x=482, y=600)

root.mainloop()