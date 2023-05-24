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

model = load_model('model/alphabets4.h5', custom_objects=custom_objects)

#Fonts
font1=("Arial Rounded MT Bold", 16)
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
        self.canvas.create_rectangle(0, 0, 660, 540, fill=bg1) #Main Frame
        self.canvas.create_rectangle(780, 0, 1400, 680, fill=bg1)
        self.canvas.create_rectangle(0, 550, 660, 800, fill=bg1) #Detection
        
        # Capture Button
        self.cap_btn_status = False
        self.cap_btn_text = tkinter.StringVar()
        self.cap_btn_text.set("Capture")
        self.cap_btn = tkinter.Button(window, textvariable=self.cap_btn_text, font=font2, bg=bg2, fg=fg1, width=15, height=1, command=self.toggle_capture_button)

        # Add The Button to Canvas
        self.canvas.create_window(380, 750, window=self.cap_btn)
        
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

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:

            # Set up the bbox
            bbox_size = (60, 60)
            bbox = [(int(self.vid.width // 2 - bbox_size[0] // 2), int(self.vid.height // 2 - bbox_size[1] // 2)), 
                    (int(self.vid.width // 2 + bbox_size[0] // 2), int(self.vid.height // 2 + bbox_size[1] // 2))]
            
            img_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
            img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, (28, 28))

            # Draw bbox on canvas
            cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 2)
            
            # Display main video frame on canvas 
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(10, 50, image = self.photo, anchor = tkinter.NW)
            self.canvas.create_text(330, 25, text="Main Frame", fill=fg1, anchor=tkinter.CENTER, font=font1)

            # Display cropped image on canvas
            self.cropped_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.resize(img_gray, (200, 200))))
            self.canvas.create_image(self.cropped_photo.width()+10, self.photo.height()+100, image = self.cropped_photo, anchor = tkinter.NE)
            self.canvas.create_text(330, 570, text="Detection Result", fill=fg1, anchor=tkinter.CENTER, font=font1)

            # Make prediction
            result, probability = self.vid.prediction(img_gray, model)
            
            # Display prediction and result on canvas
            self.canvas.delete("prediction")
            self.canvas.create_text(280, 600, text=f"Letter \t: {chr(result+65)}", fill=fg1, anchor=tkinter.NW, font=font2, tag="prediction")
            
            self.canvas.delete("result")
            self.canvas.create_text(280, 640, text=f"Probability: {probability*100:.2f}%", fill=fg1, anchor=tkinter.NW, font=font2, tag="result")

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
            self.canvas.create_text(1100, 25, text="Image Result", fill=fg1, anchor=tkinter.CENTER, font=font1)
            if result in img_dict and self.cap_btn_status:
                img = cv2.imread(img_dict[result])

                # display cropped image on canvas
                self.img_result = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.resize(img, (540, 540))))
                self.canvas.create_image(self.photo.width()*2+80, self.photo.height()+115, image = self.img_result, anchor = tkinter.SE)
                print(chr(result+65))

            # retrieve file name and print it on canvas
            if result in self.img_names and self.cap_btn_status:
                if self.img_name_id is not None:
                    self.canvas.delete(self.img_name_id)
                self.img_name_id = self.canvas.create_text(1100, 640, text=self.img_names[result], fill=fg1, anchor=tkinter.CENTER, font=font1)
                img = cv2.imread(self.img_names[result])
                
        self.window.after(self.delay, self.update)

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        
        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
        
    def prediction(self, image, model):
        img = cv2.resize(image, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        predict = model.predict(img)
        prob = np.amax(predict)
        class_index = np.argmax(predict, axis=1)
        result = class_index[0]
        if prob < 0.75:
            result = 0
            prob = 0
        return result, prob
    
    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")