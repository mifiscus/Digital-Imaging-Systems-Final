# Digital-Imaging-Systems-Final

Final project for digital imaging systems class, where I created a GUI (Tkinter) and implemented several image processing effects to test on a selected image. The GUI displays the original and processed image, with several buttons and sliders for the user to choose from.

The most complex processing feature is the multi-level thresholding technique, which separates the image into different layers, separated by "threshold" values determined by the algorithm. This algorithm can be explained more in detail here: https://doi.org/10.1016/j.patrec.2007.09.005. On top of this, the user can choose to separately color each of the layers generated by the algorithm, resulting in interesting images. Examples can be seen below using lena.bmp and town.jpg:

![Capture](https://user-images.githubusercontent.com/60052720/112931558-3ebbed00-90d1-11eb-9a5d-83c6df3345dd.PNG)
![lena_t5](https://user-images.githubusercontent.com/60052720/112927310-9eae9580-90c9-11eb-9f25-43f8881c036f.PNG)
![town_alleffects](https://user-images.githubusercontent.com/60052720/112927288-99e9e180-90c9-11eb-99d8-3d643f5b4e29.PNG)
