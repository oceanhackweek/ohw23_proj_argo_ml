import streamlit as st
from PIL import Image
from argo_plot import lib

st.markdown("<h1 style='text-align: center;'>TESTE</h1>", unsafe_allow_html=True)

"OHW"

pnboia_img = Image.open('images/logo_pnboia.png')

st.image([pnboia_img], caption=None, width=50, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

lib.plot_map()
