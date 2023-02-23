import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from numpy import load
import matplotlib.pyplot as plt
import random
from utils import distance, group_by_clusters, get_similar
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import homogeneity_score, completeness_score, contingency_matrix

st.set_page_config(page_title="Product Recommender", layout="wide",
                   initial_sidebar_state="auto", menu_items=None, page_icon="random")

@st.cache_data
def load_data():
    df = pd.read_csv("product_images.csv")
    images = df.to_numpy()
    X_tsne = load('X_tsne.npy')
    labels = load('labels.npy')
    centres = load('centres.npy')
    true_labels = pd.read_csv("true_label.csv").to_numpy()
    true_labels = np.array([each[0] for each in true_labels])
    return labels, centres, images, X_tsne, true_labels


@st.cache_data
def initial_display():
    return list(range(len(images)))


@st.cache_data
def get_metrics():
    h_score=round(homogeneity_score(true_labels, labels), 4)
    c_score = round(completeness_score(true_labels, labels), 4)
    s_score = round(silhouette_score(X_tsne, labels), 4)
    return h_score, c_score, s_score

# Initialization
if 'key' not in st.session_state:
    st.session_state['key'] = 'value'

# Session State also supports attribute based syntax
if 'key' not in st.session_state:
    st.session_state.key = 'value'

st.session_state.page=0

st.title('Image-based Product Recommendation')

# States

sample_idx = None

labels, centres, images, X_tsne, true_labels = load_data()
h_score, c_score, s_score = get_metrics()

# print(initial_ids)
initial_ids = initial_display()
groups = group_by_clusters(labels)

def img(i):
    pixels = [255 - each for each in images[i]]
    return np.reshape(pixels, (28,28))


def get_similar_wrapped(sample_index, n_similar):
    return get_similar(X=X_tsne, labels=labels, groups=groups, sample_index=sample_idx, n_similar=n_similar)


tab1, tab2 = st.tabs(["Top Picks", "Model Evaluation"])

# st.balloons()

with tab1:
    col1,col2=st.columns(2)

    with col1:
        st.subheader("Choose products to view similar recommendations")
        # showmore = st.button(label="Show more", type="primary")
        # if showmore:
        #     st.session_state.order = random.sample(list(range(len(images))), 15)

        pixel_data = np.random.randint(256, size=(28, 28))

        subcols = st.columns(3)
        col_height=5
        for i in range(3):
            with subcols[i]:
                for idx in initial_ids[i*col_height:i*col_height+col_height]:
                    st.image(img(idx), width=100)
                    sample_button = st.button(label="See more like this",
                                    key=idx)
                    if sample_button:
                        sample_idx = idx

    with col2:
        st.write(
            "Select a product to get recommendations")
        recom_no = st.slider('Number of Recommendations', 0, 50, 10, 5)
        if sample_idx is not None:
            recoms = st.subheader("If you like... ")
            st.image(img(sample_idx), width=150)
            st.write("–"*30)
            st.subheader("you might also like...")
            subcols = st.columns(3)
            recom_lst = get_similar_wrapped(
                sample_index=sample_idx, n_similar=recom_no)
            col_content=[
                [v for i, v in enumerate(recom_lst) if i % 3 == 0],
                [v for i, v in enumerate(recom_lst) if (i-1) % 3 == 0],
                [v for i, v in enumerate(recom_lst) if (i-2) % 3 == 0]
            ]

            for i in range(3):
                with subcols[i]:
                    for idx in col_content[i]:
                        st.image(img(idx), width=100)
                        sample_button = st.button(label="See more like this",
                                        key=idx)
                        if sample_button:
                            sample_idx = idx

with tab2:
    st.subheader("Evaluation Metrics")
    st.write("Homogeneity Score")
    st.write(h_score)
    st.write("Completeness Score")
    st.write(c_score)
    st.write("Silhouette Score")
    st.write(s_score)
