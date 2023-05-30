import tkinter
import cv2
import numpy as np
import PIL.Image, PIL.ImageTk
import time
from tensorflow.keras.models import load_model

model = load_model('model/digits.h5')

class App:
    def __init__(self, window, window_title, video_source=1):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source
        self.vid = MyVideoCapture(video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=1920, height=1080)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot = tkinter.Button(window, text = "Snapshot", width = 50, command = self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        # 
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame =self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.COLOR_RGB2BGR)

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
            img_gray = cv2.resize(img_gray, (200, 200))
            
            # Draw bbox on canvas
            cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 2)
            
            # Display main video frame on canvas 
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

            # Display cropped image on canvas
            self.cropped_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img_gray))
            self.canvas.create_image(self.canvas.winfo_width()-self.photo.width(), 0, image = self.cropped_photo, anchor = tkinter.NE)

            # Make prediction
            result, probability = self.vid.prediction(img_gray, model)
            
            # Display prediction and result on canvas
            # self.canvas.delete("prediction")
            # self.canvas.delete("result")
            self.canvas.create_text(10, 10, text=f"Result: {result}", fill="yellow", anchor=tkinter.NW, font=("Helvetica", 16), tag="prediction")
            self.canvas.create_text(10, 40, text=f"Probability: {probability:.2f}", fill="yellow", anchor=tkinter.NW, font=("Helvetica", 16), tag="result")

            # create dictionary to store image file names and result values
            img_dict = {
                0: 'image0.jpg',
                1: 'image1.jpg',
                2: 'image2.jpg',
                3: 'image3.jpg',
                4: 'image4.jpg',
                5: 'image5.jpg',
                6: 'image6.jpg',
                7: 'image7.jpg',
                8: 'image8.jpg',
                9: 'image9.jpg'
                }
            
            # retrieve file name and print result based on input result
            if result in img_dict:
                img = cv2.imread(img_dict[result])

                # Display cropped image on canvas
                self.img_result = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.resize(img, (300, 300))))
                self.canvas.create_image(self.canvas.winfo_width()-self.img_result.width(), self.cropped_photo.height()+400, image = self.img_result, anchor = tkinter.SE)

                print(result)

        self.window.after(self.delay, self.update)

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)

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
        class_index = model.predict_classes(img)
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