import streamlit as st
import sounddevice as sd
import wavio as wv
import time
import os
import Functions

st.header("Spoken Language Identifier")
st.write("It can identify given languages:")
st.write("\n1. Marathi\n2. English\n3. Punjabi\n4. Odia\n5. Hindi\n6. Urdu\n7. Assamese\n8. Malayalam")
st.write("\n\n")

upload_file = st.file_uploader("Select audio file", type=['mp3'])

if upload_file is not None:

    with open(os.path.join("fileDir", upload_file.name), "wb") as f:
        f.write((upload_file).getbuffer())

    audio = upload_file.read()
    st.audio(audio)

    if st.button('Predict'):
        file_path = os.path.join('fileDir', upload_file.name)
        X = Functions.get_features(file_path)
        OUTPUT = Functions.prediction(X)
        st.header(f'Prediction: {OUTPUT}')
