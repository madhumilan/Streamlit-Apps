import os
import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

st.title("Cats v/s Dogs Classifier")

loaded_model = tf.keras.models.load_model('my_model')

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


if __name__ == '__main__':
    # Select a file
    st.write("Select an image to test the classifier.")
    if st.checkbox('Click to select'):
        folder_path = st.text_input('Test Images Folder', 'test_images')
#         if st.checkbox('Change directory'):
#             st.write("Enter test_images and hit Enter")
#             folder_path = st.text_input('Test Images Folder', 'test_images')
        filename = file_selector(folder_path=folder_path)
        img_name = filename[filename.index('/'):]
        st.write('You selected `%s`' % img_name[1:])
        st.write('Choose a different image using the drop down')

        img = image.load_img(filename, target_size=(150,150,3))
        x = image.img_to_array(img)
        x = np.expand_dims(img, axis=0)

        output = loaded_model.predict(x)
        st.image(img, caption="Selected image")
        if(output < 0.5):
            st.write("It is a cat.")
        else:
            st.write("It is a dog.")
