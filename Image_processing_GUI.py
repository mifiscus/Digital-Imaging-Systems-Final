from tkinter import filedialog
import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import math

# global variables
MARGIN = 10  # px
MAXDIM = 530

class App():
    def __init__(self, window, window_title, image_path="lena.bmp"):
        self.chosen_color = [0, 0, 0]   # Initialize chosen color RGB array
        np.seterr(divide='ignore', invalid='ignore')   # Ignore divide by zero errors
        self.window = window
        self.window.title(window_title)

        # Load image using OpenCV
        self.cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.NEWcv_img = self.cv_img.copy()  # for recursive processing
        self.height, self.width, num_channel = self.cv_img.shape

        # Create a Frame that can fit the images
        self.frame1 = tk.Frame(self.window , width=100, height=100, bg='white')  # size not important
        self.frame1.pack(fill=tk.BOTH)

        # Create two Canvases for image display
        self.canvas0 = tk.Canvas(self.frame1, width=MAXDIM, height=MAXDIM+(3*MARGIN), bg='blue')  # original
        self.canvas0.pack(side=tk.LEFT)
        self.canvas1 = tk.Canvas(self.frame1, width=MAXDIM, height=MAXDIM+(3*MARGIN), bg='orange')  # original
        self.canvas1.pack()

        # PhotoImage
        self.photoOG = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))  # original
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))  # modified

        # Add PhotoImage to Canvas
        self.canvas0.create_image(MAXDIM//2, MAXDIM//2, image=self.photoOG)
        self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo)

        # Write caption for both images
        self.canvas0.create_text(MAXDIM//2, MAXDIM+(MARGIN), text="Original Photo", font="Tahoma 20")
        self.canvas1.create_text(MAXDIM//2, MAXDIM+(MARGIN), text="Modified Photo", font="Tahoma 20")

# ##############################################################################################
# ################################   PARAMETER TOOLBAR   #######################################
# ##############################################################################################

        # Create a Frame that can fit the images
        self.frame2 = tk.Frame(self.window , width=200, height=400, bg='white')
        self.frame2.pack(side=tk.BOTTOM, fill=tk.BOTH)

        # Create Label for GUI description text
        self.label = tk.Label(self.frame2, text="Michael Fiscus Final Project", font="Times 25 bold")
        self.label.pack(anchor=tk.W)
        self.label = tk.Label(self.frame2, text="Reset to return to original, must reset after every threshold, all effects reset blur", font="Times 16")
        self.label.pack(anchor=tk.W)

        # Create a Button for resetting the image
        self.button_res = tk.Button(self.frame2, text="Reset", command=self.reset)
        self.button_res.place(x=10, y=80)

        # Create a Button for negating the image
        self.button_neg = tk.Button(self.frame2, text="Negative", command=self.negative)
        self.button_neg.place(x=10, y=110)

        # Create a Scale that lets users blur the image
        self.scale_blur = tk.Scale(self.frame2, label="Blur", orient=tk.HORIZONTAL, command=self.blur)
        self.scale_blur.place(x=10, y=140)

        #Create buttons to finalize the blur level of the image
        self.button_blur = tk.Button(self.frame2, text="Set Blur", command=self.blur_set)
        self.button_blur.place(x=10, y=200)


        # Create buttons to apply different n-levels (2, 4, 6, 8) of multi-level thresholding
        self.button_thresh2 = tk.Button(self.frame2, text="Threshold 3", command=lambda: self.threshold(2))
        self.button_thresh2.place(x=150, y=80)

        self.button_thresh4 = tk.Button(self.frame2, text="Threshold 5", command=lambda: self.threshold(4))
        self.button_thresh4.place(x=150, y=110)

        self.button_thresh6 = tk.Button(self.frame2, text="Threshold 7", command=lambda: self.threshold(6))
        self.button_thresh6.place(x=150, y=140)

        self.button_thresh8 = tk.Button(self.frame2, text="Threshold 9", command=lambda: self.threshold(8))
        self.button_thresh8.place(x=150, y=170)


        # Create a scale that lets users select a color layer to change with the color button
        self.scale_layer = tk.Scale(self.frame2, label="layer",orient=tk.HORIZONTAL, from_ = 1, to = 9)
        self.scale_layer.place(x=250, y=80)

        # Create 3 scales to allow user to
        self.scale_red = tk.Scale(self.frame2, label="red", orient=tk.HORIZONTAL, to=255, command=self.color_view_red)
        self.scale_red.place(x=250, y=140)
        self.scale_green = tk.Scale(self.frame2, label="green", orient=tk.HORIZONTAL, to=255, command=self.color_view_green)
        self.scale_green.place(x=250, y=200)
        self.scale_blue = tk.Scale(self.frame2, label="blue", orient=tk.HORIZONTAL, to=255, command=self.color_view_blue)
        self.scale_blue.place(x=250, y=260)

        # Create a button to color the selected layer using the selected RGB values from the scales
        self.button_color = tk.Button(self.frame2, text="Color current layer", command= self.color_changer)
        self.button_color.place(x=365, y=187)

        # Space filler
        self.filler = tk.Button(self.frame2, height = 17)
        self.filler.pack(side=tk.RIGHT)

        # Create a canvas to display the chosen color that the user selects
        self.canvas2 = tk.Canvas(self.frame2, width=50, height=50, bg=self.rgb_convert())  # original
        self.canvas2.place(x=365, y=80)

        self.window.mainloop()

##############################################################################################
#################################  CALLBACK FUNCTIONS  #######################################
##############################################################################################
    # Callback for reset button
    def reset(self):
        self.NEWcv_img = self.cv_img
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.NEWcv_img))
        self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo)

    # Callback for negative button
    def negative(self):
        self.NEWcv_img = 255 - self.NEWcv_img
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.NEWcv_img))
        self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo)

    # Callback for blur scale
    def blur(self, k):
        k = self.scale_blur.get()  # value of slider
        self.NEWcv_img_blur = cv2.blur(self.NEWcv_img, (int(k/2+1), int(k/2+1)))
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.NEWcv_img_blur))
        self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo)

    def blur_set(self):
        self.NEWcv_img = self.NEWcv_img_blur
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.NEWcv_img))
        self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo)

    # return weighted mean of pixel range
    def weighted_mean(self, img, a, b):
        # Only accepts up to 8 thresholds
        if (a > b):
            mean = 0
            return mean

        p1 = (img >= a)
        p2 = (img <= b)
        X = np.multiply(p1, p2)
        Y = np.multiply(img, X)
        sum_ = np.sum(X)
        mean = np.sum(Y) / sum_
        return mean

    # uses Multilevel thresholding paper algorithm to segment image into n parts
    def threshold(self, n):
        if (self.NEWcv_img.ndim != 2):
            img = cv2.cvtColor(self.NEWcv_img, cv2.COLOR_BGR2GRAY)   # only works when image in grayscale

        # Step 2: set image range initially to [0, 255]
        a = 0
        b = 255
        k1 = 0.65   # free parameter (from trial and error)
        k2 = 0.85   # free parameter (from trial and error)
        T = []   # new threshold list
        threshold_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)

        # make sure u is defined in case n = 2 and for loop does not run
        p1 = (img >= a)
        p2 = (img <= b)
        X = np.multiply(p1, p2)
        Y = np.multiply(img, X)
        u = np.sum(Y) / np.sum(X)   # mean

        # Step 1: repeat steps 2-6 (n/2-1) times, where n is number of thresholds
        for i in range(int(n / 2 - 1)):

            # Step 3: calculate mean and standard deviation for all pixels in image
            p1 = (img >= a)
            p2 = (img <= b)
            X = np.multiply(p1, p2)
            Y = np.multiply(img, X)
            u = np.sum(Y) / np.sum(X)   # mean

            Z = Y - u
            Z = np.multiply(Z, X)
            W = np.multiply(Z, Z)
            sigma = math.sqrt(np.sum(W) / np.sum(X))   # standard deviation

            # Step 4: calculate sub-range boundaries
            T1 = u - k1 * sigma
            T2 = u + k2 * sigma

            # Step 5: assign threshold values to pixels within [a, T1] and [T2, b] to respective weighted means
            thresh1 = self.weighted_mean(img, a, T1)
            thresh2 = self.weighted_mean(img, T2, b)
            T.append(thresh1)
            T.append(thresh2)

            # Step 6: adjust sub-range boundaries
            a = T1 + 1
            b = T2 - 1

        # Step 7: repeat step 5 with T1 = u, T2 = u + 1
        T1 = u
        T2 = u + 1
        thresh1 = self.weighted_mean(img, a, T1)
        thresh2 = self.weighted_mean(img, T2, b)
        T.append(thresh1)
        T.append(thresh2)
        T.sort()   # final list of n-thresholds
        # End of Mutli-level Thresholding Paper Algorithm
        print(T)
        # scans through image and checks if pixel in in between certain thresholds, then changes value accordingly
        for k in range (n):
            threshold_img[img >= T[k]] = int((255 / n) * (k + 1))
        threshold_img[img <= T[0]] = 0   # beginning case

        # creates boolean array of masks for the color changer to decide to change each level of pixel
        self.mask = np.zeros((n+1, threshold_img.shape[0], threshold_img.shape[1]), dtype=bool)
        for q in range (n):
            self.mask[(q + 1), threshold_img == int((255 / n) * (q + 1))] = True
        self.mask[0, threshold_img == 0] = True

        self.NEWcv_img = threshold_img
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.NEWcv_img))
        self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo)

        self.scale_layer = tk.Scale(self.frame2, label="layer",orient=tk.HORIZONTAL, from_ = 1, to = n+1)
        self.scale_layer.place(x=250, y=80)

    def color_changer(self):
        color_img = np.zeros((self.NEWcv_img.shape[0], self.NEWcv_img.shape[1], 3), np.uint8)

        # Check if current NEWcv_img is 2 dimensional or 3 dimensional to avoid errors
        if (self.NEWcv_img.ndim != 2):
            color_img = self.NEWcv_img
        else:
            color_img[:, :, 0] = self.NEWcv_img
            color_img[:, :, 1] = self.NEWcv_img
            color_img[:, :, 2] = self.NEWcv_img

        # Changes the image based on the mask selected using the layer slider, using the user selected color
        layer = self.scale_layer.get() - 1
        color_img[self.mask[layer] == True] = self.chosen_color

        self.NEWcv_img = color_img
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.NEWcv_img))
        self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo)

    # Retrieves R slider value and updates display canvas
    def color_view_red(self, red):
        red = self.scale_red.get()
        self.chosen_color[0] = red
        self.canvas2 = tk.Canvas(self.frame2, width=50, height=50, bg=self.rgb_convert())  # original
        self.canvas2.place(x=365, y=80)

    # Retrieves G slider value and updates display canvas
    def color_view_green(self, green):
        green = self.scale_green.get()
        self.chosen_color[1] = green
        self.canvas2 = tk.Canvas(self.frame2, width=50, height=50, bg=self.rgb_convert())  # original
        self.canvas2.place(x=365, y=80)

    # Retrieves B slider value and updates display canvas
    def color_view_blue(self, blue):
        blue = self.scale_blue.get()
        self.chosen_color[2] = blue
        self.canvas2 = tk.Canvas(self.frame2, width=50, height=50, bg=self.rgb_convert())  # original
        self.canvas2.place(x=365, y=80)

    # converts RGB value to Tkinter color strings
    def rgb_convert(self):
        return f'#{self.chosen_color[0]:02x}{self.chosen_color[1]:02x}{self.chosen_color[2]:02x}'

##############################################################################################
# Create a window and pass it to the Application object
App(tk.Tk(), "Final Project")
