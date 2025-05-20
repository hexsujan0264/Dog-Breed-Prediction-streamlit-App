#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


#Loading the Model
model = load_model('dog_breed.h5')

#Name of Classes
CLASS_NAMES = ['Scottish Deerhound','Maltese Dog','Bernese Mountain Dog']

#Setting Title of App
st.title("Dog Breed Prediction")
st.markdown("Upload an image of the dog")

#Uploading the dog image
dog_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])  # Allow jpg and jpeg formats
submit = st.button('Predict')
reset = st.button('Reset')  # Reset button

# On predict button click
if submit:
    if dog_image is not None:
        try:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            # Displaying the image
            st.image(opencv_image, channels="BGR")
            # Resizing the image
            opencv_image = cv2.resize(opencv_image, (224, 224))
            # Normalize the image data
            opencv_image = opencv_image / 255.0  # Normalize to [0, 1]
            # Convert image to 4 Dimension
            opencv_image = opencv_image.reshape(1, 224, 224, 3)
            # Make Prediction
            Y_pred = model.predict(opencv_image)
            confidence = np.max(Y_pred)  # Get the confidence score
            breed = CLASS_NAMES[np.argmax(Y_pred)]

            st.title(f"The Dog Breed is {breed} with a confidence of {confidence:.2f}")
        except Exception as e:
            st.error("Error processing the image: " + str(e))  # Display error message
    else:
        st.warning("Please upload an image.")  # Warning if no image is uploaded

# On reset button click
if reset:
    st.experimental_rerun()  # Reset the app
