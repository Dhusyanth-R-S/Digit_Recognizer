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
.warning {
    font-size: 28px;
    font-weight: bold;
    color: #e67e22;
    text-align: center;
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
        if st.button("D R A W"):
            st.session_state.show_canvas = True
            st.rerun()

    else:
        if st.session_state.prediction is not None:
            if isinstance(st.session_state.prediction, int):
                st.markdown(
                    f"<div class='prediction'>{st.session_state.prediction}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='warning'>{st.session_state.prediction}</div>",
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
            if st.button("Clear"):
                st.session_state.canvas_key = f"canvas_{np.random.randint(1_000_000)}"
                st.session_state.prediction = None
                st.rerun()

        with c2:
            if st.button("Predict"):
                if canvas.image_data is None:
                    st.stop()

                img = canvas.image_data[:, :, :3].mean(axis=2).astype(np.uint8)

                digit_28 = Image.fromarray(img).resize(
                    (28, 28),
                    resample=Image.BILINEAR
                )

                X = np.array(digit_28, dtype=np.uint8).reshape(1, -1)

                probs = model.predict_proba(X)[0]
                best_prob = probs.max()
                pred = probs.argmax()

                if best_prob < 0.6:
                    st.session_state.prediction = "Please redraw"
                else:
                    st.session_state.prediction = int(pred)

                st.rerun()
