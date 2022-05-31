import os
import glob
import requests

import numpy as np
from PIL import Image
import streamlit as st

from venomai import unet, predictor, preprocess

st.set_page_config(
     page_title="AI-assisted Haemorrhage Analysis",
     page_icon="",
    #  layout="wide",
     initial_sidebar_state="expanded"
 )

@st.cache
def download_models():
    if len(glob.glob('models/*.ckpt')) != 5:
        print('Downloading models...')
        for i in range(5):
            URL = f'https://github.com/laprade117/venom-ai-haemorrhage-analysis-tool/releases/latest/download/unet_final_{i}.ckpt'
            response = requests.get(URL)
            filename = f'models/unet_final_{i}.ckpt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            open(f'models/unet_final_{i}.ckpt', 'wb').write(response.content)

if __name__ == '__main__':
    
    download_models()

    st.title('AI-assisted Haemorrhage Analysis')

    # uploaded_file = st.sidebar.file_uploader("hello") 
    uploaded_file = st.sidebar.file_uploader("Upload an image with template", type=['.jpg','.png','.tif'], accept_multiple_files=False)
    
    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        image = np.array(image)
        image = preprocess.preprocess_image(image)

        final_predictions = None
        for i in range(5):
            model = unet.UNet.load_from_checkpoint(f'models/unet_final_{i}.ckpt')
            predictions, windows = predictor.predict_image(model, image, apply_preprocessing=False)
            if i == 0:
                final_predictions = predictions
            else:
                final_predictions += predictions
        predictions = final_predictions / 5
        
        haus, real_areas, luminance_values, mean_rgb_values = predictor.compute_haemorrhagic_units(predictions, windows, return_stats=True)
        masks = np.round((predictions > 0.5)[:,:,:,None] * mean_rgb_values[:,None,None,:]).astype('uint8')

        windows = list(np.array(windows, dtype=object))
        masks = list(np.array(masks, dtype=object))

        captions_hau = [f'HaU: {haus[i]:.02f}' for i in range(4)]
        captions = [f'Area: {real_areas[i]:.02f} mm\N{SUPERSCRIPT TWO}\nLuminance: {luminance_values[i]:.02f}' for i in range(4)]

        col1, col2, col3 = st.columns(3)
        col1.metric("HaU", f"{np.nanmean(haus):.02f}")
        col2.metric("Area", f"{np.nanmean(real_areas):.02f} mm\N{SUPERSCRIPT TWO}")
        col3.metric("Luminance", f"{np.nanmean(luminance_values):.02f}")

        st.image(windows, caption=captions_hau, width=174, clamp=[0,255])
        st.image(masks, caption=captions, width=174, clamp=[0,255])
        st.image(image)