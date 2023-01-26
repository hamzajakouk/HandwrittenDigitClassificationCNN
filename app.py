import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# load the pre-trained model
model = load_model('/content/drive/MyDrive/training_dataset/Model/fire_and_smoke_model.h5')

# function to classify the image
def classify():
    # get the file path of the image
    filepath = filedialog.askopenfilename()
    # load the image and preprocess it
    img = image.load_img(filepath, target_size=(256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    # classify the image
    pred = model.predict(img)
    # check the predicted class
    if pred[0][0] > 0.5:
        result_label.config(text="Fire")
    else:
        result_label.config(text="Smoke")

# create the GUI
root = tk.Tk()
root.title("Fire and Smoke Classifier")

# create the button to select the image
select_button = tk.Button(root, text="Select Image", command=classify)
select_button.pack()

# create the label to display the result
result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()