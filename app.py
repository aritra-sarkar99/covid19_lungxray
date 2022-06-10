import streamlit as st
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


# File id: 1WdSuKJfguWX8KcYnVIYjZzT-9tQOE1Jp

model = tf.keras.models.load_model("resnet_model.h5")

def load_image(image_file):
    img = Image.open(image_file)
    return img

def save_image(image_file):
    with open(image_file.name,"wb") as f:
        f.write(image_file.getbuffer())
    st.success('File saved')

def predict(imageName):
    img_path = "D:\MyProjects\covid19_lungxray\\" + imageName
    img = image.load_img(img_path, target_size = (128, 128))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_batch = img_batch * (1./255)
    covid =  model.predict(img_batch)[0][0]* 100
    normal =  model.predict(img_batch)[0][1]* 100
    pneumonia =  model.predict(img_batch)[0][2]* 100

    d = {
        "covid": "%.2f" % covid,
        "normal": "%.2f" % normal,
        "pneumonia":"%.2f" % pneumonia
    }

    return d

def piechart(d):
    labels = 'Covid', 'Normal', 'Pneumonia'
    sizes = [float(d["covid"]), float(d["normal"]), float(d["pneumonia"])]
    explode = (0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels,explode=explode , colors=["#F52A2A", "#28B463", "#F1C40F"], autopct='%1.2f%%',
             startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)




st.header('Detect Lung disease from Chest X-Ray')


image_file = st.file_uploader('Upload X-ray image', type=['png','jpg','jpeg'])
col1, col2 = st.columns(2)
if image_file:
    st.image(load_image(image_file),width=250)
    save_image(image_file)
    # print(image_file.name)
    d = {}
    with st.spinner('Predicting...'):
        d = predict(image_file.name)
    st.subheader("Covid: " + d["covid"] + " %")
    st.subheader("Normal: " + d["normal"] + " %")
    st.subheader("Pneumonia: " + d["pneumonia"] + " %")

    if float(d["covid"])>30:
        st.error('The person might be infected with COVID-19')
    elif float(d["pneumonia"])>30:
        st.error('The person might be infected with Viral Pneumonia')

    with st.spinner('Generating chart...'):
        piechart(d)
    os.remove(image_file.name)



