import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="✍️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("digit_logreg_pipeline.pkl")

model = load_model()

if "show_canvas" not in st.session_state:
    st.session_state.show_canvas = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_1"

st.markdown("""
<style>
.prediction {
    font-size: 64px;
    font-weight: bold;
    color: #2ecc71;
    text-align: center;
    margin-bottom: 10px;
}
.draw-btn button {
    font-size: 48px !important;
    height: 120px;
    width: 100%;
}
.predict-btn button {
    font-size: 28px !important;
    height: 70px;
    width: 100%;
}
.clear-btn button {
    font-size: 14px !important;
    height: 35px;
}
</style>
""", unsafe_allow_html=True)

left, right = st.columns([1, 2])

with right:

    if not st.session_state.show_canvas:
        st.markdown('<div class="draw-btn">', unsafe_allow_html=True)
        if st.button("D R A W"):
            st.session_state.show_canvas = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        if st.session_state.prediction is not None:
            st.markdown(
                f"<div class='prediction'>{st.session_state.prediction}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown("<div class='prediction'> </div>", unsafe_allow_html=True)

        canvas = st_canvas(
            fill_color="rgba(255,255,255,1)",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key=st.session_state.canvas_key
        )

        c1, c2 = st.columns([1, 4])

        with c1:
            st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
            if st.button("Clear"):
                st.session_state.canvas_key = f"canvas_{np.random.randint(0, 1_000_000)}"
                st.session_state.prediction = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
            if st.button("Predict"):
                if canvas.image_data is not None:
                    img = canvas.image_data[:, :, 0]
                    img = Image.fromarray(img).resize((28, 28))
                    img = np.array(img)
                    img = img / 255.0
                    img = img.reshape(1, -1)
                    pred = model.predict(img)[0]
                    st.session_state.prediction = int(pred)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
