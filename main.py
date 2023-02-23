import streamlit as st
import pandas as pd
import numpy as np
from numpy import load
import random
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import homogeneity_score, completeness_score, contingency_matrix

st.set_page_config(page_title="Product Recommendation", layout="wide",
                   initial_sidebar_state="expanded", menu_items=None, page_icon="random")

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
    ids = list(range(len(images)))
    random.shuffle(ids)
    return ids


@st.cache_data
def get_metrics():
    h_score=round(homogeneity_score(true_labels, labels), 4)
    c_score = round(completeness_score(true_labels, labels), 4)
    s_score = round(silhouette_score(X_tsne, labels), 4)
    return h_score, c_score, s_score


@st.cache_data
def group_by_clusters(labels):
    mapper = {}
    for i in range(len(set(labels))):
        mapper[i] = [k for k, v in enumerate(labels) if v == i]
    for k, v in mapper.items():
        random.shuffle(v)
    return mapper


@st.cache_data
def make_plots():
    legend = ["tshirt/top", "trousers", "pullover", "dress",
              "coat", "sandal", "shirt", "sneaker", "bag", "ankleboots"]

    our_clusters, z = plt.subplots(figsize=(20, 10))
    x_vals = [each[0] for each in X_tsne]
    y_vals = [each[1] for each in X_tsne]

    for k in np.unique(labels):
        x_vals = [X_tsne[i][0] for i in range(len(X_tsne)) if labels[i] == k]
        y_vals = [X_tsne[i][1] for i in range(len(X_tsne)) if labels[i] == k]
        z.scatter(x_vals, y_vals, label=k)
    z.legend()

    x_vals = [each[0] for each in X_tsne]
    y_vals = [each[1] for each in X_tsne]

    true_clusters, j = plt.subplots(figsize=(20, 10))

    for k in np.unique(true_labels):
        x_vals = [X_tsne[i][0]
                  for i in range(len(X_tsne)) if true_labels[i] == k]
        y_vals = [X_tsne[i][1]
                  for i in range(len(X_tsne)) if true_labels[i] == k]
        j.scatter(x_vals, y_vals, label=f"{k}-{legend[k]}")
    j.legend()
    return our_clusters, true_clusters

def distance(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2)))


def get_similar(X, labels, groups, sample_index, n_similar):
    sample_label = labels[sample_index]
    distances = [
        (i, distance(X[sample_index], X[i])) for i in groups[sample_label]
    ]
    sorted_distances = sorted(distances, key=lambda x: x[1])
    n_nearest = [x[0] for x in sorted_distances[1:n_similar+1]]
    return n_nearest

# Session State also supports attribute based syntax
if 'page' not in st.session_state:
    st.session_state.page = 0

if 'sample_idx' not in st.session_state:
    st.session_state.sample_idx = -1

st.title('Image-based Product Recommendation')

# States


labels, centres, images, X_tsne, true_labels = load_data()
h_score, c_score, s_score = get_metrics()

initial_ids = initial_display()
groups = group_by_clusters(labels)




display_size = 20

def img(i):
    pixels = [255 - each for each in images[i]]
    return np.reshape(pixels, (28,28))


def get_similar_wrapped(sample_index, n_similar):
    return get_similar(X=X_tsne, labels=labels, groups=groups, sample_index=sample_index, n_similar=n_similar)

st.write("idx state", st.session_state.sample_idx)

tab1, tab2, tab3, tab4 = st.tabs(["Top Picks", "Browse by Category", "Model Evaluation", "Acknowledgements"])
with tab1:
    st.write("idx state", st.session_state.sample_idx)
    # col1,col2=st.columns(2)
    st.subheader("Choose products to view similar recommendations")
    # with col1:
    button_layout = st.columns(7)
    with button_layout[0]:
        prevpage = st.button(label="< Prev Page", type="primary")
        if prevpage and st.session_state.page > 0:
            st.session_state.page -= display_size
    with button_layout[1]:
        nextpage = st.button(label="Next Page >", type="primary")
        if nextpage and st.session_state.page < 10000/display_size:
            st.session_state.page += display_size



    subcols = st.columns(4)
    col_height=5
    for i in range(4):
        with subcols[i]:
            for idx in initial_ids[st.session_state.page:st.session_state.page+display_size][i*col_height:i*col_height+col_height]:
                st.image(img(idx), width=100, caption=idx)
                sample_button1 = st.button(label="See more like this",
                                key=idx)
                if sample_button1:
                    st.session_state.sample_idx = idx



with tab2:
    st.header("View products by category")
    for j, title in enumerate(["Ankleboots & Sandals","Bags","Dresses & Tops","T-shirt & Tops",
                               "Sneakers & Sandals","Coats, Pullovers & Tops","Trousers","Pullovers & Tops"]):
        with st.expander(str(title)):
            subcols = st.columns(5)
            col_height = 4
            for i in range(5):
                with subcols[i]:
                    for idx in groups[j][:display_size][st.session_state.page:st.session_state.page+display_size][i*col_height:i*col_height+col_height]:
                        st.image(img(idx), width=100, caption=idx)
                        sample_button1 = st.button(label="See more like this",
                                                key="cat"+str(idx))
                        if sample_button1:
                            st.session_state.sample_idx = idx



st.sidebar.markdown(
    "## Select a product to get recommendations")
st.sidebar.write("idx state", st.session_state.sample_idx)
recom_no = st.sidebar.slider('Number of Recommendations', 0, 50, 10, 5)
if st.session_state.sample_idx >= 0:
    st.sidebar.markdown("# If you like... ")
    st.sidebar.image(img(st.session_state.sample_idx), width=200,
                        caption=st.session_state.sample_idx)
    st.sidebar.markdown("# you might also like...")
    subcols = st.sidebar.columns(3)
    recom_lst = get_similar_wrapped(
        sample_index=st.session_state.sample_idx, n_similar=recom_no)
    col_content = [
        [v for i, v in enumerate(recom_lst) if i % 3 == 0],
        [v for i, v in enumerate(recom_lst) if (i-1) % 3 == 0],
        [v for i, v in enumerate(recom_lst) if (i-2) % 3 == 0]
    ]

    for i in range(3):
        with subcols[i]:
            for idx in col_content[i]:
                st.image(img(idx), width=100, caption=idx)
                sample_button2 = st.button(label="See more like this",
                                            key=idx+9999)
                if sample_button2:
                    st.session_state.sample_idx = idx

with tab3:
    st.subheader("Model")
    st.write("Feature Extraction method: t-SNE")
    st.write("Number of clusters k: 8")
    st.write("Initialisation Method: k-means++")
    st.subheader("Evaluation Metrics")
    st.write("Homogeneity Score")
    st.write("– A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class.")
    st.write("Score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling.")
    st.write(h_score)
    st.write("Completeness Score")
    st.write("– A clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster.")
    st.write("Score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling.")
    st.write(c_score)
    st.write("Silhouette Score")
    st.write("– A measure of how similar an object is to its own cluster(cohesion) compared to other clusters(separation)")
    st.write("The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.")
    st.write(s_score)

with tab4:
    st.header("Acknowledgements")
    st.balloons()
    st.markdown("### *Thank you* Deyu!")
    st.write()
    # our_clusters, true_clusters = make_plots()

    # st.subheader("Clustering of pre-processed data")
    # st.pyplot(our_clusters)


    # st.subheader("Clustering using true labels")
    # st.pyplot(true_clusters)

