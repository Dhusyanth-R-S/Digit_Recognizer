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
/* center draw section */
.draw-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 40px;
}

/* text above button */
.draw-text {
    font-size: 18px;
    margin-bottom: 20px;
}

/* BIG circular button */
button[data-testid="stButton"] {
    width: 200px;
    height: 200px;
    border-radius: 50%;
    font-size: 32px !important;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

if not st.session_state.show_canvas:
    st.markdown("<div class='draw-container'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='draw-text'>Click draw to make me recognize what you draw!</div>",
        unsafe_allow_html=True
    )
    if st.button("DRAW"):
        st.session_state.show_canvas = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

else:
    if st.session_state.prediction is not None:
        st.markdown(
            f"<h1 style='text-align:center;'>{st.session_state.prediction}</h1>",
            unsafe_allow_html=True
        )

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
