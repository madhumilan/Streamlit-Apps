import os
import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

'''
path = "C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/"
dataset = os.path.join(path,"cats-and-dogs/")
print(dataset)
'''
path = os.getcwd()
st.write(path)
loaded_model = tf.keras.models.load_model(path)

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


if __name__ == '__main__':
    # Select a file
    if st.checkbox('Select a file in current directory'):
        folder_path = '.'
        if st.checkbox('Change directory'):
            folder_path = st.text_input('Enter folder path', '.')
        filename = file_selector(folder_path=folder_path)
        st.write('You selected `%s`' % filename)

        img = image.load_img(filename, target_size=(150,150,3))
        x = image.img_to_array(img)
        x = np.expand_dims(img, axis=0)

        output = loaded_model.predict(x)
        st.image(img, caption="Selected image")
        if(output < 0.5):
            st.write("It is a cat.")
        else:
            st.write("It is a dog.")
