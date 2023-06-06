import tkinter
import cv2
import numpy as np
import PIL.Image, PIL.ImageTk
import time
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D

# Define a dictionary that maps DepthwiseConv2D to Conv2D
custom_objects = {'DepthwiseConv2D': Conv2D}

model = load_model('model/digits.h5', custom_objects=custom_objects)

#Fonts
font1=("Arial Rounded MT Bold", 14)
font2=("Helavetica",16)

#Backgrounds
bg1="#1D4B4A"
bg2="#54B2AF"

#Foregrounds
fg1="#FFFFFF"

class App:
    
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.window.iconbitmap('icon.ico')
        self.video_source = video_source
        
        # open video source
        self.vid = MyVideoCapture(video_source)
        self.img_name_id = None

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=1920, height=1080)
        self.canvas.pack(fill = "both", expand = True)

        self.img_names = {
            0: 'Lumba 4 Lateral Kontras',
            1: 'Lumbal 4 AP Kontras',
            2: 'Lumbal 4 Lateral',
            3: 'Lumbal 4 Obliqe',
            4: 'Lumbal 4 Target',
            5: 'Lumbal 5 AP Kontras',
            6: 'Lumbal 5 AP',
            7: 'Lumbal 5 Lateral Kontras',
            8: 'Lumbal 5 Lateral',
            9: 'Lumbal 5 Oblie',
            10: 'Lumbal 5 Target',
            11: 'Lumbal 5-S1 AP Kontras',
            12: 'Lumbal 5-S1 AP',
            13: 'Lumbal 5-S1 Lateral Kontras',
            14: 'Lumbal 5-S1 Lateral',
            15: 'Lumbar 4 AP'
        }

        # self.canvas.create_rectangle(100,200,1000,600, outline="#E8F9FD", fill= 'black')
        self.canvas.create_rectangle(0, 0, 1920, 1080, fill=bg1)
        self.canvas.create_rectangle(20, 20, 720, 780, fill=bg1, outline=bg2, width=2) #Main Frame
        self.canvas.create_rectangle(750, 20, 1510, 780, fill=bg1, outline=bg2, width=2)
        # self.canvas.create_rectangle(0, 550, 660, 800, fill=bg1, outline=bg2, width=2) #Detection
        
        # Capture Button
        self.cap_btn_status = False
        self.cap_btn_text = tkinter.StringVar()
        self.cap_btn_text.set("Capture")
        self.cap_btn = tkinter.Button(window, textvariable=self.cap_btn_text, font=font2, bg=bg2, fg=fg1, width=10, height=1, command=self.toggle_capture_button)

        # Camera Button
        self.cam_status = False
        self.cam_text = tkinter.StringVar()
        self.cam_text.set("CAM-1")
        self.cam = tkinter.Button(window, textvariable=self.cam_text, font=font2, bg=bg2, fg=fg1, width=8, height=1, command=self.toggle_cam_button)

        
        # Add The Button to Canvas
        self.canvas.create_window(590, 750, window=self.cap_btn)
        self.canvas.create_window(110, 80, window=self.cam)
        
        ####
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame =self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.COLOR_RGB2BGR)

    def toggle_capture_button(self):
        self.cap_btn_status = not self.cap_btn_status
        if self.cap_btn_status:
            self.cap_btn_text.set("Capturing...")
        else:
            self.cap_btn_text.set("Capture")

    def toggle_cam_button(self):
        self.cam_status = not self.cam_status
        if self.cam_status:
            self.cam_text.set("CAM-2")
            self.vid.set_video_source(1)  # Set video_source to 1 for CAM-2
        else:
            self.cam_text.set("CAM-1")
            self.vid.set_video_source(0)  # Set video_source to 0 for CAM-1

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            # Set up the bbox
            bbox_size = (60, 60)
            bbox1 = [(int(self.vid.width // 2 - bbox_size[0]), int(self.vid.height // 2 - bbox_size[1] // 2)),
                     (int(self.vid.width // 2), int(self.vid.height // 2 + bbox_size[1] // 2))]

            bbox2 = [(int(self.vid.width // 2), int(self.vid.height // 2 - bbox_size[1] // 2)),
                     (int(self.vid.width // 2 + bbox_size[0]), int(self.vid.height // 2 + bbox_size[1] // 2))]

            img_cropped1 = frame[bbox1[0][1]:bbox1[1][1], bbox1[0][0]:bbox1[1][0]]
            img_cropped2 = frame[bbox2[0][1]:bbox2[1][1], bbox2[0][0]:bbox2[1][0]]

            img_gray1 = cv2.cvtColor(img_cropped1, cv2.COLOR_BGR2GRAY)
            img_gray1 = cv2.resize(img_gray1, (28, 28))

            img_gray2 = cv2.cvtColor(img_cropped2, cv2.COLOR_BGR2GRAY)
            img_gray2 = cv2.resize(img_gray2, (28, 28))

            # Draw bboxes on canvas
            cv2.rectangle(frame, bbox1[0], bbox1[1], (255, 0, 0), 2)
            cv2.rectangle(frame, bbox2[0], bbox2[1], (0, 255, 0), 2)

            # Display main video frame on canvas
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(50, 50, image=self.photo, anchor=tkinter.NW)
            self.canvas.create_text(350, 35, text="Camera", fill=fg1, anchor=tkinter.CENTER, font=font1)

            # Display cropped gray images on canvas
            self.canvas.create_text(350, 555, text="Detection Result", fill=fg1, anchor=tkinter.CENTER, font=font1)

            self.cropped_photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.resize(img_gray1, (200, 200))))
            self.canvas.create_image(self.cropped_photo1.width() + 50, self.photo.height() + 90, image=self.cropped_photo1,
                                     anchor=tkinter.NE)

            self.cropped_photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.resize(img_gray2, (200, 200))))
            self.canvas.create_image(self.cropped_photo2.width() + 250, self.photo.height() + 90, image=self.cropped_photo2,
                                    anchor=tkinter.NE)

            # Make predictions
            result1, probability1 = self.vid.prediction(img_gray1, model)
            result2, probability2 = self.vid.prediction(img_gray2, model)
            
            if result1 == 0:
                result = result2
            else:
                result = int(str(result1) + str(result2))


            # Display predictions and results on canvas
            
            self.canvas.delete("merah")
            self.canvas.create_text(490, 600, text=f"Digit 1\t: {result1}", fill=fg1, anchor=tkinter.NW, font=font2,
                                    tag="merah")
            
            self.canvas.delete("hijau")
            self.canvas.create_text(490, 640, text=f"Digit 2\t: {result2}", fill=fg1, anchor=tkinter.NW, font=font2,
                                    tag="hijau")
            
            self.canvas.delete("prediction")
            self.canvas.create_text(490, 680, text=f"Result\t: {result}", fill=fg1, anchor=tkinter.NW, font=font2,
                                    tag="prediction")

            # self.canvas.delete("result")
            # self.canvas.create_text(420, 640, text=f"Probability: {(probability1+probability2) * 0.5 * 100:.2f}%", fill=fg1, anchor=tkinter.NW,
            #                         font=font2, tag="result")


            # create dictionary to store image file names and result values
            img_dict = {
                0: 'bones/Lumba 4 Lateral Kontras.jpg',
                1: 'bones/lumbal 4 AP Kontras.jpg',
                2: 'bones/lumbal 4 Lateral.jpg',
                3: 'bones/Lumbal 4 Obliqe .jpg',
                4: 'bones/Lumbal 4 target.jpg',
                5: 'bones/Lumbal 5 AP kontras.jpg',
                6: 'bones/Lumbal 5 AP.jpg',
                7: 'bones/Lumbal 5 lateral kontras.jpg',
                8: 'bones/Lumbal 5 lateral.jpg',
                9: 'bones/Lumbal 5 Oblie.jpg',
                10: 'bones/Lumbal 5 target.jpg',
                11: 'bones/Lumbal 5-S1 Ap kontras.jpg',
                12: 'bones/Lumbal 5-S1 Ap.jpg',
                13: 'bones/Lumbal 5-S1 lateral kontras.jpg',
                14: 'bones/Lumbal 5-S1 Lateral.jpg',
                15: 'bones/lumbar 4 Ap.jpg'
                }
            
            # retrieve file name and print result based on input result
            self.canvas.create_text(1125, 35, text="Scan Result", fill=fg1, anchor=tkinter.CENTER, font=font1)
            if result in img_dict and self.cap_btn_status:
                img = cv2.imread(img_dict[result])

                # display cropped image on canvas
                self.img_result = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.resize(img, (680, 680))))
                self.canvas.create_image(self.photo.width()*2+190, self.photo.height()+250, image = self.img_result, anchor = tkinter.SE)
                print(result)

            # retrieve file name and print it on canvas
            if result in self.img_names and self.cap_btn_status:
                if self.img_name_id is not None:
                    self.canvas.delete(self.img_name_id)
                self.img_name_id = self.canvas.create_text(1125, 750, text=self.img_names[result], fill=fg1, anchor=tkinter.CENTER, font=font1)
                img = cv2.imread(self.img_names[result])
                
        self.window.after(self.delay, self.update)

class MyVideoCapture:
    def __init__(self, video_source=0):
        self.video_source = video_source
        self.vid = None
        self.open_video_source()

    def open_video_source(self):
        self.vid = cv2.VideoCapture(self.video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.vid.isOpened():
            print(f"Unable to open video source {self.video_source}.")
            self.vid = None

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) if self.vid is not None else 640
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) if self.vid is not None else 480

    def get_frame(self):
        if self.vid is not None:
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            # Return a black screen if the camera is not found
            return (False, np.zeros((self.height, self.width, 3), dtype=np.uint8))

    def prediction(self, image, model):
        img = cv2.resize(image, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        predict = model.predict(img)
        prob = np.amax(predict)
        class_index = model.predict_classes(img)
        result = class_index[0]
        if prob < 0.75:
            result = 0
            prob = 0
        return result, prob

    def __del__(self):
        if self.vid is not None:
            self.vid.release()

    def set_video_source(self, video_source):
        self.video_source = video_source
        self.open_video_source()


# Create a window and pass it to the Application object
App(tkinter.Tk(), "C-Arm Simulator")