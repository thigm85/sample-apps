import os
import clip
import time
import streamlit as st
import shutil
from math import floor
from PIL import Image
from urllib.request import urlretrieve

from vespa.application import Vespa
from vespa.query import QueryModel, ANN, QueryRankingFeature, RankProfile as Ranking
from embedding import (
    translate_model_names_to_valid_vespa_field_names,
    TextProcessor,
    decode_string_to_media,
)

st.set_page_config(layout="wide")

VESPA_URL = os.environ.get("VESPA_ENDPOINT", "http://localhost:8080")
VESPA_CERT_PATH = os.environ.get("VESPA_CERT_PATH", None)
with open("certificate.txt", "w") as f:
    f.write(VESPA_CERT_PATH)


app = Vespa(
    url=VESPA_URL,
    cert="certificate.txt",
)


@st.cache(ttl=24 * 60 * 60)
def get_available_clip_model_names():
    return clip.available_models()


@st.cache(ttl=7 * 24 * 60 * 60)
def get_text_processor(clip_model_name):
    text_processor = TextProcessor(model_name=clip_model_name)
    return text_processor


@st.cache
def download_photos():
    urlretrieve(
        "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
        "data.zip",
    )
    shutil.unpack_archive("data.zip", "photos", "zip")
    return "photos"


def get_image(image_file_name, image_dir):
    return Image.open(os.path.join(image_dir, image_file_name))


def vespa_query(query, clip_model_name):
    vespa_model_name = translate_model_names_to_valid_vespa_field_names(clip_model_name)
    image_vector_name = vespa_model_name + "_image"
    text_vector_name = vespa_model_name + "_text"
    ranking_name = vespa_model_name + "_similarity"
    text_processor = get_text_processor(clip_model_name=clip_model_name)
    result = app.query(
        query=query,
        query_model=QueryModel(
            name=vespa_model_name,
            match_phase=ANN(
                doc_vector=image_vector_name,
                query_vector=text_vector_name,
                hits=100,
                label="clip_ann",
            ),
            rank_profile=Ranking(name=ranking_name, list_features=False),
            query_properties=[
                QueryRankingFeature(name=text_vector_name, mapping=text_processor.embed)
            ],
        ),
        **{"presentation.timing": "true"}
    )
    return [hit["fields"]["image_file_name"] for hit in result.hits], result.json[
        "timing"
    ]


photos_dir = download_photos()
IMG_FOLDER = photos_dir

clip_model_name = st.sidebar.selectbox(
    "Select CLIP model", get_available_clip_model_names()
)

out1, col1, out2 = st.columns([3, 1, 3])
col1.image("https://docs.vespa.ai/assets/logos/vespa-logo-full-black.svg", width=100)
query_input = st.text_input(label="", value="a man surfing", key="query_input")

start = time.time()
image_file_names, timing = vespa_query(query=query_input, clip_model_name=clip_model_name)
placeholder = st.empty()
number_rows = floor(len(images) / 3)
remainder = len(images) % 3
if number_rows > 0:
    for i in range(number_rows):
        col1, col2, col3 = st.columns(3)
        col1.image(get_image(image_file_names[3*i], image_dir=IMG_FOLDER))
        col2.image(get_image(image_file_names[3*i + 1], image_dir=IMG_FOLDER))
        col3.image(get_image(image_file_names[3*i + 2], image_dir=IMG_FOLDER))
if remainder > 0:
    cols = st.columns(3)
    for i in range(remainder):
        cols[i].image(get_image(image_file_names[3*number_rows+i], image_dir=IMG_FOLDER))
total_timing = time.time() - start
vespa_search_time = round(timing["searchtime"], 2)
total_time = round(total_timing, 2)
other_time = round(total_time - vespa_search_time, 2)
placeholder.write(
    "**Vespa search time: {}s**. Network related time: {}s. Total time: {}s".format(
        vespa_search_time, other_time, total_time
    )
)
