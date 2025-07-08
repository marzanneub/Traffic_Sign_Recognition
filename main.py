import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

class_name = ['Ahead only',
 'Beware of icesnow',
 'Bicycles crossing',
 'Bumpy road',
 'Children crossing',
 'Dangerous curve to the left',
 'Dangerous curve to the right',
 'Double curve',
 'End no passing veh  3.5 tons',
 'End of Speed Limit 80',
 'End of no passing',
 'End speed + passing limits',
 'General caution',
 'Go straight or left',
 'Go straight or right',
 'Keep left',
 'Keep right',
 'No entry',
 'No passing',
 'No passing for vechiles over 3.5 metric tons',
 'No vechiles',
 'Pedestrians',
 'Priority road',
 'Right-of-way at the next intersection',
 'Road narrows on the right',
 'Road work',
 'Roundabout mandatory',
 'Slippery road',
 'Speed Limit 100',
 'Speed Limit 120',
 'Speed Limit 20',
 'Speed Limit 30',
 'Speed Limit 50',
 'Speed Limit 60',
 'Speed Limit 70',
 'Speed Limit 80',
 'Stop',
 'Traffic signals',
 'Turn left ahead',
 'Turn right ahead',
 'Vechiles over 3.5 metric tons prohibited',
 'Wild animals crossing',
 'Yield']


st.header("Traffic Sign Recognition")
test_image = st.file_uploader("Upload the sign: ")
if test_image is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(test_image, use_container_width=True, caption="Uploaded sign")
    with col2:
        st.write("### Prediction Result")
        result_index = model_prediction(test_image)
        st.success(f"Model is recognizing it as: **{class_name[result_index]}**")

