from io import StringIO
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import streamlit as st
import pandas as pd
from functionforDownloadButtons import download_button
import torch

def _max_width_():
    max_width_str = f"max-width: 1800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}

    </style>    
    """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_icon="images/icon.png", page_title="Zero Shot")



c2, c3 = st.columns([6, 1])

with c2:
    c31, c32 = st.columns([12, 2])
    with c31:
        st.caption("")
        st.title("Shot 0")
    with c32:
        st.image(
            "images/logo.png",
            width=200,
        )


# Convert PDF to JPG

uploaded_file = st.file_uploader(
    " ",
    key="1",
    help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
)


result = ""
list_keywords = []
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='')
    form = st.form(key="annotation")
    with form:
        question_input = st.text_input("Enter your query here")
        #list_keywords = question_input.split(',')
        submitted = st.form_submit_button(label="Submit")

    if uploaded_file is not None and submitted is not None:
        
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        #url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        #image = Image.open(requests.get(url, stream=True).raw)
        
        #asd = question_input.split(',')
        #st.write(question_input.split(','))
        inputs = processor(text=question_input.split(','), images=image, return_tensors="pt", padding=True)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        max_idx, max_val = max(enumerate(probs[0].tolist()), key=lambda x: x[1])
        st.write(max_idx, max_val)
