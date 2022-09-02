import tkinter as tk
from PIL import Image, ImageTk
from joblib import load
import torch

from src.cnn.cnn import CNN
from cv2 import imread
import numpy as np
from tkinter import filedialog
from src.feature_fusion.feature_vector_generation import get_patch_yi

with torch.no_grad():
    our_cnn = CNN()
    our_cnn.load_state_dict(torch.load('../src/Cnn.pt',
                                       map_location=lambda storage, loc: storage))
    our_cnn.eval()
    our_cnn = our_cnn.double()

svm_model = load('../src/Svm.pt')

window = tk.Tk()

canvas = tk.Canvas(window, width=600, height=300)
canvas.grid(columnspan=3, rowspan=3)

logo = Image.open('../logo.png')
logo = ImageTk.PhotoImage(logo)
logo_label = tk.Label(image=logo)
logo_label.image = logo
logo_label.grid(column=1, row=0)

instructions = tk.Label(window, text="Select an image to run the detection test.", font="Calibri")
instructions.grid(columnspan=3, column=0, row=1)


def get_feature_vector(image_path: str, model):
    feature_vector = np.empty((1, 400))
    feature_vector[0, :] = get_patch_yi(model, imread(image_path))
    return feature_vector


def open_file():
    global file
    file = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                      filetype=(("jpeg files", "*.jpg"), ("all files", "*.*")))

    img = Image.open(file)
    img = img.resize((384, 256), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(img)

    label2 = tk.Label(image=photo)
    label2.image = photo
    label2.grid(column=1, row=4)


def detect():
    image_path = file
    image_feature_vector = get_feature_vector(image_path, our_cnn)
    res = svm_model.predict(image_feature_vector)
    res_label = tk.Label()
    res_label.config(text=" ")
    res_label.grid(columnspan=3, column=1, row=5)
    if res == 0:
        res_label.config(text="Image is not Tampered.", font=15)
    else:
        res_label.config(text="Image is Tampered.", font=15)


browse_text = tk.StringVar()
browse_btn = tk.Button(window, textvariable=browse_text, command=lambda: open_file(), font="Calibri")
browse_text.set("Browse Image")
browse_btn.grid(column=1, row=2)

detect_text = tk.StringVar()
detect_btn = tk.Button(window, textvariable=detect_text, command=lambda: detect(), font="Calibri")
detect_text.set("Detect")
detect_btn.grid(column=1, row=8)

window.mainloop()
