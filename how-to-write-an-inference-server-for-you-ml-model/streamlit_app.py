import streamlit as st
from qqp_inference.model import PythonPredictor

predictor = PythonPredictor.create_for_demo()
q1 = st.text_input("Fist question",)
q2 = st.text_input("Second question",)
st.write(predictor.predict(payload={"q1": q1, "q2": q2}))
