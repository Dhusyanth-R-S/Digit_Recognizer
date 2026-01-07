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

st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.top-hint {
    text-align: center;
    font-size: 18px;
    color: #cfcfcf;
    margin-bottom: 18px;
}

.draw-circle {
    display: flex;
    justify-content: center;
}
.draw-circle button {
    width: 160px;
    height: 160px;
    border-radius: 50%;
    font-size: 26px !important;
    font-weight: 700;
}

.prediction {
    font-size: 72px;
    font-weight: 800;
    color: #2ecc71;
    text-align: center;
    animation: slide 0.5s ease-out;
}

@keyframes slide {
    0% { transform: translateY(-20px); opacity: 0; }
    100% { transform: translateY(0); opacity: 1; }
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
    align-items: center;
    gap: 12px;
}

.clear-btn button {
    height: 36px;
    width: 90px;
    font-size: 14px !important;
}

.predict-btn button {
    height: 70px;
    width: 100%;
    font-size: 26px !important;
    font-weight: 800;
    background-color: #7bed9f !important;
    color: #000000 !important;
}

@media (min-width: 768px) {
    .layout {
        display: flex;
        gap: 40px;
        align-items: center;
    }
    .prediction {
        text-align: left;
        font-size: 88px;
    }
}
</style>
""", unsafe_allow_html=True)

if not st.session_state.show_canvas:
    st.markdown("<div class='top-hint'>Click draw to make me recognize what you draw!</div>", unsafe_allow_html=True)
    st.markdown("<div class='draw-circle'>", unsafe_allow_html=True)
    if st.button("DRAW"):
        st.session_state.show_canvas = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='layout'>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        if st.session_state.prediction is not None:
            if isinstance(st.session_state.prediction, int):
                st.markdown(f"<div class='prediction'>{st.session_state.prediction}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='warning'>{st.session_state.prediction}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='prediction'>&nbsp;</div>", unsafe_allow_html=True)

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

        c1, c2 = st.columns([1, 3])
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
                digit_28 = Image.fromarray(img).resize((28, 28), resample=Image.BILINEAR)
                X = np.array(digit_28, dtype=np.uint8).reshape(1, -1)
                probs = model.predict_proba(X)[0]
                best_prob = probs.max()
                pred = probs.argmax()
                if best_prob < 0.6:
                    st.session_state.prediction = "Please redraw"
                else:
                    st.session_state.prediction = int(pred)
                st.rerun()

    with right:
        pass

    st.markdown("</div>", unsafe_allow_html=True)
