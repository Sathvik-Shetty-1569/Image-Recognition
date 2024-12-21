from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np

# Tkinter GUI Window Configuration
from tkinter import *
from tkinter import font,Label,ttk,filedialog
from PIL import Image, ImageTk

SHOP_NAME = "OBJECT/IMAGE DETECTING PROJECT"

window =Tk()
img =""
window.title(SHOP_NAME)
win_width= window.winfo_screenwidth()    
half_width = win_width//2
win_height= window.winfo_screenheight()         
half_height = win_height//2      
window.geometry("%dx%d" % (half_width, half_height))
window.config(bg="light grey")


class_labels = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

f=Path("model2_structure.json")
model2_structure=f.read_text()
model = model_from_json(model2_structure)
model.load_weights("model2.weights.h5")

def scan_image(path):
    import matplotlib.pyplot as plt
    from tensorflow.keras.utils import load_img, img_to_array



    img = load_img(path,target_size = (32,32))
    plt.imshow(img)

    from tensorflow.keras.utils import img_to_array
    image_to_test = img_to_array(img)
    list_of_images=np.expand_dims(image_to_test,axis = 0)
    results = model.predict(list_of_images)
    single_result = results[0]
    most_likely_class_index=int(np.argmax(single_result))
    class_likelihood = single_result[most_likely_class_index]
    class_label = class_labels[most_likely_class_index]
    print("This image is a {} likelihood: {:2f}".format(class_label, class_likelihood))
    res = "This Image is Classified as a "+class_label
    resultLabel.config(text=res)
    resultLabel.pack(side=BOTTOM,pady = 30)

def open_filechooser():
    global img
    filepath = filedialog.askopenfilename()
    print(filepath)
    scan_image(filepath)
    image = Image.open(filepath)
    image = image.resize((200, 200)) 
    img= ImageTk.PhotoImage(image)
    imgLabel.pack(side=TOP,pady=30)
    imgLabel.config(image=img)
    imgLabel.image=img
    

# TITLE FRAME 
title_frame = Frame(master=window,bg="light grey")
title_frame.pack(side=TOP)

title = Label(master=title_frame,text=SHOP_NAME, fg="red",bg="light grey", font=('Franklin Gothic Medium', 20))
title.pack()

image_path = r"C:\Users\SATHVIK\OneDrive\Documents\PROJECT\Image Recognition\Screenshot 2024-06-13 190035.png"
image = ImageTk.PhotoImage(Image.open(image_path).resize((100,50)))

imgLabel = Label(window,image=img)
imgLabel.pack_forget()

resultLabel = Label(window,text="",fg="green")
resultLabel.pack_forget()

button = Button(window,text="",image=image,command=open_filechooser,bg="light grey")
button.pack(side=TOP)

window.mainloop()
