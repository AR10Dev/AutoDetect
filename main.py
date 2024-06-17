import customtkinter
from customtkinter import *
from ultralytics import YOLO
import cv2
import math
from PIL import Image, ImageTk
import PIL
import threading
import queue

IMG_PATH = None
VIDEO_PATH = None
MODEL_PATH = "Yolo/yolov8n.pt"


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


class AutoDetectorApp(CTk):
    def __init__(self):
        super().__init__()

        # Set window title and disable resizing
        self.video_detect_button = None
        self.video_path = None
        self.video_error_path_header = None
        self.video_path_header = None
        self.image_detect_button = None
        self.image_path = None
        self.image_error_path_header = None
        self.image_path_header = None
        self.title("AutoDetector v1.0.0")
        self.resizable(False, False)

        # Create tab view
        self.tab_view = Tabs(master=self, width=800, height=600)
        self.tab_view.grid(row=0, column=0, padx=20, pady=20)

        # Initialize Image and Video tab elements
        self.initialize_image_tab_elements()
        self.initialize_video_tab_elements()

    def initialize_image_tab_elements(self):
        self.image_path_header = CTkLabel(master=self.tab_view.tab("Image"), text="", font=("Arial", 20))
        self.image_path_header.place(relx=0.5, rely=0.6, anchor="center")

        self.image_error_path_header = CTkLabel(master=self.tab_view.tab("Image"), text="", font=("Arial", 20),
                                                text_color="red")
        self.image_error_path_header.place(relx=0.5, rely=0.6, anchor="center")

        self.image_path = CTkLabel(master=self.tab_view.tab("Image"), text="", font=("Arial", 15))
        self.image_path.place(relx=0.5, rely=0.65, anchor="center")

        self.image_detect_button = CTkButton(master=self.tab_view.tab("Image"), text="Detect now", font=("Arial", 30),
                                             command=lambda: image_detection(), width=150, height=18,
                                             corner_radius=10)
        self.image_detect_button.place_forget()

    def initialize_video_tab_elements(self):
        self.video_path_header = CTkLabel(master=self.tab_view.tab("Video"), text="", font=("Arial", 20))
        self.video_path_header.place(relx=0.5, rely=0.6, anchor="center")

        self.video_error_path_header = CTkLabel(master=self.tab_view.tab("Video"), text="", font=("Arial", 20),
                                                text_color="red")
        self.video_error_path_header.place(relx=0.5, rely=0.6, anchor="center")

        self.video_path = CTkLabel(master=self.tab_view.tab("Video"), text="", font=("Arial", 15))
        self.video_path.place(relx=0.5, rely=0.65, anchor="center")

        self.video_detect_button = CTkButton(master=self.tab_view.tab("Video"), text="Detect now", font=("Arial", 30),
                                             command=lambda: video_detection(), width=150, height=18,
                                             corner_radius=10)
        self.video_detect_button.place_forget()


class Tabs(CTkTabview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # Create tabs
        self.add("Home")
        self.add("Image")
        self.add("Video")
        self.add("Webcam")

        # Home tab content
        home_title = CTkLabel(master=self.tab("Home"), text="Welcome to", font=("Arial", 40))
        home_title.place(relx=0.5, rely=0.1, anchor="center")
        home_subtitle = CTkLabel(master=self.tab("Home"), text="AutoDetector", font=("Arial", 60), text_color="#007bff")
        home_subtitle.place(relx=0.5, rely=0.2, anchor="center")
        home_description = CTkLabel(master=self.tab("Home"),
                                    text="This is a simple object detection application that can detect\nobjects in "
                                         "images, videos, and webcam\n\n\n\nUse the tabs to navigate the application.",
                                    font=("Arial", 20))
        home_description.place(relx=0.5, rely=0.5, anchor="center")
        home_footer = CTkLabel(master=self.tab("Home"),
                               text="Developed by: Luca Facchini (LF-D3v) & Avaab Razzaq (AR10Dev)", font=("Arial", 13))
        home_footer.place(relx=0.5, rely=0.9, anchor="center")

        # Image tab content
        image_title = CTkLabel(master=self.tab("Image"), text="Detection from", font=("Arial", 30))
        image_title.place(relx=0.5, rely=0.1, anchor="center")
        image_subtitle = CTkLabel(master=self.tab("Image"), text="Image", font=("Arial", 40), text_color="#007bff")
        image_subtitle.place(relx=0.5, rely=0.2, anchor="center")

        image_select_file_button = CTkButton(master=self.tab("Image"), text="Select Image", font=("Arial", 30),
                                             command=lambda: open_file_dialog('Image'), width=150, height=18,
                                             corner_radius=10)
        image_select_file_button.place(relx=0.5, rely=0.4, anchor="center")

        # Video tab content
        video_title = CTkLabel(master=self.tab("Video"), text="Detection from", font=("Arial", 30))
        video_title.place(relx=0.5, rely=0.1, anchor="center")
        video_subtitle = CTkLabel(master=self.tab("Video"), text="Video", font=("Arial", 40), text_color="#007bff")
        video_subtitle.place(relx=0.5, rely=0.2, anchor="center")

        video_select_file_button = CTkButton(master=self.tab("Video"), text="Select Video", font=("Arial", 30),
                                             command=lambda: open_file_dialog('Video'), width=150, height=18,
                                             corner_radius=10)
        video_select_file_button.place(relx=0.5, rely=0.4, anchor="center")

        # Webcam tab content
        webcam_title = CTkLabel(master=self.tab("Webcam"), text="Detection from", font=("Arial", 30))
        webcam_title.place(relx=0.5, rely=0.1, anchor="center")
        webcam_subtitle = CTkLabel(master=self.tab("Webcam"), text="Webcam", font=("Arial", 40), text_color="#007bff")
        webcam_subtitle.place(relx=0.5, rely=0.2, anchor="center")

        webcam_start_button = CTkButton(master=self.tab("Webcam"), text="Start Webcam", font=("Arial", 30),
                                             command=lambda: webcam_detection(), width=150, height=18,
                                             corner_radius=10)
        webcam_start_button.place(relx=0.5, rely=0.4, anchor="center")




def open_file_dialog(selection):
    global IMG_PATH
    global VIDEO_PATH

    if selection == 'Image':
        IMG_PATH = filedialog.askopenfilename()
        print_file_path(selection, IMG_PATH)
    elif selection == 'Video':
        VIDEO_PATH = filedialog.askopenfilename()
        print_file_path(selection, VIDEO_PATH)



def print_file_path(selection, file_path):
    if selection == 'Image':
        # check if the file format is valid.
        if file_path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            app.image_path_header.configure(text="Currently selected image", text_color="green")
            app.image_error_path_header.configure(text="")

            app.image_path.configure(text=file_path)
            app.image_detect_button.place(relx=0.5, rely=0.85, anchor="center")
        elif file_path == "":
            app.image_error_path_header.configure(text="No file selected", text_color="red")
            app.image_path_header.configure(text="")
            app.image_path.configure(text="")
            app.image_detect_button.place_forget()
        else:
            app.image_error_path_header.configure(text="Invalid file format", text_color="red")
            app.image_path_header.configure(text="")
            app.image_path.configure(text="")
            app.image_detect_button.place_forget()

    elif selection == 'Video':
        # check if the file format is valid.
        if file_path.endswith(('.mp4', '.avi', '.mov', '.flv', '.wmv', '.mkv')):
            app.video_path_header.configure(text="Currently selected video", text_color="green")
            app.video_error_path_header.configure(text="")

            app.video_path.configure(text=file_path)
            app.video_detect_button.place(relx=0.5, rely=0.85, anchor="center")
        elif file_path == "":
            app.video_error_path_header.configure(text="No file selected", text_color="red")
            app.video_path_header.configure(text="")
            app.video_path.configure(text="")
            app.video_detect_button.place_forget()
        else:
            app.video_error_path_header.configure(text="Invalid file format", text_color="red")
            app.video_path_header.configure(text="")
            app.video_path.configure(text="")
            app.video_detect_button.place_forget()

def cv2_to_imagetk():
    image = cv2.imread(IMG_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imagePIL = PIL.Image.fromarray(image)
    imageTk = ImageTk.PhotoImage(image = imagePIL)

    return imageTk

def image_detection():
    # check if the file still exists, or if the file got changed
    if not os.path.exists(IMG_PATH):
        app.image_error_path_header.configure(text="Image not found. Did it get deleted or moved?", text_color="red")
        app.image_path_header.configure(text="")
        app.image_path.configure(text="")
        app.image_detect_button.place_forget()
        return

    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Hide the main window
    app.withdraw()

    # Create a new window to display the detected image
    img_window = customtkinter.CTkToplevel()
    img_window.title("Detected Image")
    img_window.resizable(False, False)
    img_window.minsize(720, 1080)  # Set minimum size to 720x480
    img_window.maxsize(1240, 1780)  # Set maximum size to 1240x720

    # Create the "Go Back" button and place it at the top of the window, leaving some margin
    back_button = customtkinter.CTkButton(img_window, text="Go Back")
    back_button.pack(pady=10)

    # Create a text box for detections
    detections_textbox = customtkinter.CTkTextbox(img_window, width=80, height=20, font=("Arial", 15))
    detections_textbox.pack(side="top", fill="both", expand=True)

    # Insert a title
    detections_textbox.insert("end", "Detected Objects\n\n", "title")



    # Load the image
    img = cv2.imread(IMG_PATH)
    results = model(img)

    # Process the resized image with YOLO model and draw bounding boxes
    for i, r in enumerate(results):
        boxes = r.boxes
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    # Check if there are no detections after processing all results
    if not any(results):
        detections_textbox.insert("end", "No detections found.\n", "red")
        print("Text to see if it works")
    else:
        for i, r in enumerate(results):
            boxes = r.boxes
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                detections_textbox.insert("end", f"Detection {j + 1}:\n")
                detections_textbox.insert("end", f"Class: {classNames[cls]}\n")
                detections_textbox.insert("end", f"Confidence: {confidence}\n")
                detections_textbox.insert("end", f"Coordinates: ({x1}, {y1}) - ({x2}, {y2})\n\n")



    # Resize the image proportionally
    max_width = 1280
    max_height = 720
    width, height = img.shape[1], img.shape[0]
    aspect_ratio = width / height
    if aspect_ratio > max_width / max_height:
        width = max_width
        height = int(width / aspect_ratio)
    else:
        height = max_height
        width = int(height * aspect_ratio)

    img = cv2.resize(img, (width, height))

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img_ctk = CTkImage(light_image=img_pil, dark_image=img_pil, size=(width, height))

    # Display the image in a Label and move it to the center of the window
    img_label = customtkinter.CTkLabel(img_window, image=img_ctk, text="")
    img_label.pack(fill="both", expand=True)  # Make the label fill the window

    # Keep a reference to the image to prevent garbage collection
    img_label.image = img_ctk

    # Bind the ButtonRelease event to the "Go Back" button and check if the mouse pointer is still over the button
    back_button.bind("<ButtonRelease-1>", lambda event: back(event, img_window, None, None))


def video_detection():
    # a function that simply reproduces the video in a customtkinter window.

    # Hide the main window
    app.withdraw()

    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Create a new window to display the detected video
    video_window = customtkinter.CTkToplevel()
    video_window.title("Detected Video")
    video_window.resizable(False, False)
    video_window.minsize(720, 1080)  # Set minimum size to 720x480
    video_window.maxsize(1240, 1780)  # Set maximum size to 1240x720

    # Create the "Go Back" button and place it at the top of the window, leaving some margin
    back_button = customtkinter.CTkButton(video_window, text="Go Back")
    back_button.pack(pady=10)


    # Initialize a variable to keep track of whether the video is paused
    paused = False

    # Create a text box for detections
    detections_textbox = customtkinter.CTkTextbox(video_window, width=80, height=20, font=("Arial", 15))
    detections_textbox.pack(side="top", fill="both", expand=True)

    # Insert a title
    detections_textbox.insert("end", "Detected Objects\n\n", "title")

    # Create a label to display the video
    video_label = customtkinter.CTkLabel(video_window)
    video_label.pack()

    # Create a video capture object
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Count the number of frames
    frame_number = 0;

    # Display the video
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)
        frame_number += 1

        # Update the textbox
        detections_textbox.insert("end", f"Detected objects in frame {frame_number}:\n")

        detected_objects = {}  # Dictionary to store detected objects in the frame

        for i, r in enumerate(results):
            boxes = r.boxes

            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)


                cls = int(box.cls[0])
                class_name = classNames[cls]
                if class_name not in detected_objects:
                    detected_objects[class_name] = 1
                else:
                    detected_objects[class_name] += 1

        # Append detected objects to the message
        for obj, count in detected_objects.items():
            detections_textbox.insert("end", f"{count} {obj}, ")

        # Move to the next line after appending all detected objects
        detections_textbox.insert("end", "\n\n")

        # Auto-scroll the textbox
        detections_textbox.yview_moveto(1.0)

        # Convert the image from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the image to PIL format
        img = Image.fromarray(frame)

        # Convert the image to CTkImage format
        img_ctk = CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))


        video_label.configure(image=img_ctk, text="")
        video_label.image = img_ctk


        # Update the window
        video_window.update()

        # Bind the ButtonRelease event to the "Go Back" button
        back_button.bind("<ButtonRelease-1>", lambda event: back(event, video_window, "video", cap))
    # Kill the window
    video_window.destroy()

# a simple function that starts the webcam and detects objects in real-time.
def webcam_detection():
    # close the main window
    app.withdraw()

    # Create a new window to display the webcam feed
    webcam_window = customtkinter.CTkToplevel()
    webcam_window.title("Webcam Feed")
    webcam_window.resizable(False, False)
    webcam_window.minsize(720, 720)

    # Create the "Go Back" button and place it at the top of the window, leaving some margin
    back_button = customtkinter.CTkButton(webcam_window, text="Go Back")
    back_button.pack(pady=10)

    # Create a text box for detections
    detections_textbox = customtkinter.CTkTextbox(webcam_window, width=80, height=20, font=("Arial", 15))
    detections_textbox.pack(side="top", fill="both", expand=True)

    # Insert a title
    detections_textbox.insert("end", "Detected Objects\n\n", "title")

    # Create a label to display the webcam feed
    webcam_label = customtkinter.CTkLabel(webcam_window)
    webcam_label.pack()

    # initialize the webcam and model
    cap = cv2.VideoCapture(0)
    model = YOLO(MODEL_PATH)

    while True:
        success, img = cap.read()

        # Resize the frame to fit the window
        img = cv2.resize(img, (720, 480))

        # Invert the frame
        inverted_img = cv2.flip(img, 1)
        results = model(inverted_img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # put box in cam
                cv2.rectangle(inverted_img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100

                # class name
                cls = int(box.cls[0])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(inverted_img, classNames[cls], org, font, fontScale, color, thickness)

                detections_textbox.yview_moveto(1.0)

                # Update the textbox
                detections_textbox.insert("end", f"Detected {classNames[cls]} with confidence {confidence}\n")

        # Convert the image from BGR to RGB
        inverted_img = cv2.cvtColor(inverted_img, cv2.COLOR_BGR2RGB)

        # Convert the image to PIL format
        img_pil = Image.fromarray(inverted_img)

        # Convert the image to CTkImage format
        img_ctk = CTkImage(light_image=img_pil, dark_image=img_pil, size=(img_pil.width, img_pil.height))

        webcam_label.configure(image=img_ctk, text="")
        webcam_label.image = img_ctk

        # Update the window
        webcam_window.update()

        # Bind the ButtonRelease event to the "Go Back" button
        back_button.bind("<ButtonRelease-1>", lambda event: back(event, webcam_window, "webcam", cap))

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def back(event, window, type, cap):
    if window.winfo_containing(event.x_root, event.y_root) == event.widget:
        if type == "video":
            cap.release()
            app.deiconify()
            return

        if type == "webcam":
            cap.release()

        app.deiconify()
        window.destroy()


app = AutoDetectorApp()
app.mainloop()
