import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import streamlit as st



def plot_map():

    deployment_loc = [-22.810508, -40.094289]

    m = folium.Map(location=deployment_loc, zoom_start=13)

    folium_static(m)

if __name__ == "__main__":
    plot_map()
