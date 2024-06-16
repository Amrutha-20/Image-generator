import streamlit as st
import boto3
import json
import base64
from PIL import Image
from io import BytesIO

# Function to call the Amazon Titan model for image generation
def generate_images(prompt, number_of_images, quality, height, width, cfg_scale, seed):
    bedrock_runtime = boto3.client(service_name="bedrock-runtime")
    
    payload = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,
        },
        "imageGenerationConfig": {
            "numberOfImages": number_of_images,
            "quality": quality,
            "height": height,
            "width": width,
            "cfgScale": cfg_scale,
            "seed": seed
        }
    }
    body = json.dumps(payload)
    response = bedrock_runtime.invoke_model(
        body=body,
        modelId="amazon.titan-image-generator-v1",
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get("body").read())
    images = [Image.open(BytesIO(base64.b64decode(base64_image))) for base64_image in response_body.get("images")]
    return images

# Streamlit user interface
st.title("Image Generator with Amazon Titan")
st.write("Enter a prompt and generate images using Generative AI!")

prompt = st.text_area("Enter your prompt:", value="Amazon forest tiger")
number_of_images = st.slider("Number of Images", 1, 5, 2)
quality = st.selectbox("Quality", ["standard", "premium"])
height = st.number_input("Height", min_value=64, max_value=1024, value=512, step=64)
width = st.number_input("Width", min_value=64, max_value=1024, value=512, step=64)
cfg_scale = st.slider("CFG Scale", 1.1, 10.0, 7.5)
seed = st.number_input("Seed", min_value=0, max_value=2147483647, value=0)

if st.button("Generate Images"):
    with st.spinner("Generating..."):
        images = generate_images(prompt, number_of_images, quality, height, width, cfg_scale, seed)
        st.success("Images generated!")
        
        for idx, img in enumerate(images):
            st.image(img, caption=f"Generated Image {idx + 1}")

st.write("© 2024 Generative AI Image Generator")
