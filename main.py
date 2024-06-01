import tkinter as tk
from tkinter import filedialog, messagebox  # Import messagebox
from ultralytics import YOLO
import cv2
import math
from PIL import Image, ImageTk
import os

model = YOLO("Yolo/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


def get_file_path(event):
    global PATH
    selection = selectedOpt.get()

    if selection == 'Image':
        open_file_dialog('image')
    elif selection == 'Video':
        open_file_dialog('video')

    tv_string.set(f'Selected file: {PATH}')


def open_file_dialog(mode):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    global PATH
    PATH = filedialog.askopenfilename()
    tv_string.set(f'File path: {PATH}')

    if mode == 'image':
        detect_button.config(command=lambda: image(PATH))
    elif mode == 'video':
        detect_button.config(command=lambda: video(PATH))


def image(PATH):
    # Check if the file extension is correct
    _, ext = os.path.splitext(PATH)
    if ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        messagebox.showerror("Error", "Invalid file extension. Please select a valid image file.")
        tv_string.set('Error: Invalid file extension. Please select a valid image file.')
        return
    # Close the main window
    root.withdraw()

    # Create a new window for the image
    img_window = tk.Toplevel()
    img_window.title("Image")
    img_window.resizable(False, False)  # Make the window non-resizable

    # Create a "Go Back" button
    back_button = tk.Button(img_window, text="Go Back", command=lambda: back(img_window))
    back_button.pack()

    # Load the image
    img = cv2.imread(PATH)
    results = model(img)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    # Convert the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to PIL format
    img = Image.fromarray(img)

    # Convert the image to ImageTk format
    img = ImageTk.PhotoImage(img)

    # Create a label and add the image to it
    img_label = tk.Label(img_window, image=img)
    img_label.image = img  # Keep a reference to the image
    img_label.pack(fill=tk.BOTH, expand=True)  # Make the label fill the window


def back(img_window):
    # Close the image window
    img_window.destroy()

    # Reopen the main window
    root.deiconify()


def video(PATH):
    # Check if the file extension is correct
    _, ext = os.path.splitext(PATH)
    if ext.lower() not in ['.mp4', '.avi', '.mov', '.flv', '.wmv']:
        messagebox.showerror("Error", "Invalid file extension. Please select a valid video file.")
        tv_string.set('Error: Invalid file extension. Please select a valid video file.')
        return

    cap = cv2.VideoCapture(PATH)
    close = False

    while not close:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # put box in image
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)

        cv2.imshow('Video', frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # 'q' or 'Esc' key
            break

        if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
            close = True

    cap.release()
    cv2.destroyAllWindows()


def on_enter(event):
    event.widget.config(background='gray')


def on_leave(event):
    event.widget.config(background='white')


# Main window
# Main window
root = tk.Tk()
root.geometry("800x600")  # Set window size to 800x600
root.resizable(False, False)  # Disable window resizing
root.configure(bg='dark blue')  # Set background color to dark blue
root.option_add('*Font', 'Arial 15')  # Change font to Arial and size to 15
root.option_add('*Background', 'dark blue')  # Set widget background color to dark blue
root.option_add('*Foreground', 'white')  # Set widget text color to white

# Header text
h1 = tk.Label(root, text="AutoDetect", font=("Arial", 30), bg='dark blue', fg='white')  # h1 header
h1.place(relx=0.5, rely=0.1, anchor='center')  # Center the h1 header

h3 = tk.Label(root, text="by Luca Facchini and Avaab Razzaq", font=("Arial", 20), bg='dark blue', fg='white')  # h3 header
h3.place(relx=0.5, rely=0.2, anchor='center')  # Center the h3 header

options = {'Image': 'Detection from Image', 'Video': 'Detection from Video'}

# Default option
selectedOpt = tk.StringVar()
selectedOpt.set('Image')

# Text variable
tv_string = tk.StringVar()

# Option menu
om = tk.OptionMenu(root, selectedOpt, *options.keys())
om.config(width=15)
om.place(relx=0.5, rely=0.5, anchor='center')  # Center the option menu

btn = tk.Button(root, text='Select file', bg='white', fg='black')  # Set button background color to white and text color to black
btn.place(relx=0.5, rely=0.6, anchor='center')  # Center the button
btn.bind('<Button-1>', get_file_path)
btn.bind('<Enter>', on_enter)  # Change button color to gray when mouse enters
btn.bind('<Leave>', on_leave)  # Change button color back to white when mouse leaves

# Display label for selected file path
file_label = tk.Label(root, textvariable=tv_string, bg='dark blue', fg='white')
file_label.place(relx=0.5, rely=0.7, anchor='center')  # Center the label

detect_button = tk.Button(root, text='Detect now', bg='white', fg='black')  # Create the "Detect now" button
detect_button.place(relx=0.5, rely=0.8, anchor='center')  # Center the button initially

# Start the main loop
root.mainloop()