import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", page_icon="✍️", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("digit_lr_model.pkl")

model = load_model()

if "show_canvas" not in st.session_state:
    st.session_state.show_canvas = False
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_1"
if "celebrate" not in st.session_state:
    st.session_state.celebrate = False

st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.hint {
    text-align: center;
    font-size: 18px;
    color: #bbbbbb;
    margin-bottom: 20px;
}
.draw-big button {
    font-size: 42px !important;
    height: 110px;
    width: 100%;
}
.prediction-box {
    font-size: 72px;
    font-weight: 800;
    color: #2ecc71;
    text-align: center;
    animation: pop 0.6s ease-out;
}
@keyframes pop {
    0% { transform: translateY(-30px) scale(0.7); opacity: 0; }
    100% { transform: translateY(0) scale(1); opacity: 1; }
}
.warning {
    font-size: 28px;
    font-weight: 700;
    color: #f39c12;
    text-align: center;
}
.button-row {
    display: flex;
    justify-content: space-between;
    gap: 10px;
}
.clear-btn button {
    height: 38px;
    font-size: 14px !important;
    width: 90px;
}
.predict-btn button {
    height: 60px;
    font-size: 22px !important;
    width: 100%;
    background-color: #2ecc71 !important;
    color: black !important;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

if not st.session_state.show_canvas:
    st.markdown("<div class='hint'>Click draw to make me recognize what you draw!</div>", unsafe_allow_html=True)
    if st.container().button("D R A W", key="draw", help="Start drawing"):
        st.session_state.show_canvas = True
        st.rerun()
else:
    if st.session_state.prediction is not None:
        if isinstance(st.session_state.prediction, int):
            st.markdown(f"<div class='prediction-box'>{st.session_state.prediction}</div>", unsafe_allow_html=True)
            if st.session_state.celebrate:
                st.balloons()
                st.session_state.celebrate = False
        else:
            st.markdown(f"<div class='warning'>{st.session_state.prediction}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction-box'>&nbsp;</div>", unsafe_allow_html=True)

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

    st.markdown("<div class='button-row'>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Clear", key="clear"):
            st.session_state.canvas_key = f"canvas_{np.random.randint(1_000_000)}"
            st.session_state.prediction = None
            st.rerun()
    with c2:
        if st.button("Predict", key="predict"):
            if canvas.image_data is None:
                st.stop()
            img = canvas.image_data[:, :, :3].mean(axis=2).astype(np.uint8)
            digit_28 = Image.fromarray(img).resize((28, 28), resample=Image.BILINEAR)
            X = np.array(digit_28, dtype=np.uint8).reshape(1, -1)
            probs = model.predict_proba(X)[0]
            best_prob = probs.max()
            pred = probs.argmax()
            if best_prob < 0.6:
                st.session_state.prediction = "Please redraw"
            else:
                st.session_state.prediction = int(pred)
                st.session_state.celebrate = True
            st.rerun()
