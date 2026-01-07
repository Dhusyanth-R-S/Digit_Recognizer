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
.draw-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 80px;
    text-align: center;
}

.draw-text {
    font-size: 24px;
    margin-bottom: 25px;
    color: #4b4b4b;
    font-weight: 600;
}

.draw-container div.stButton > button {
    width: 180px !important;
    height: 180px !important;
    border-radius: 50% !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    background-color: #2ecc71 !important;
    color: white !important;
    border: none !important;
}

div.stButton > button:first-child:contains("Predict") {
    background-color: #2ecc71 !important;
    color: white !important;
    font-size: 26px !important;
    padding: 15px 45px !important;
    font-weight: 800 !important;
    border: none !important;
    width: 100% !important;
}

.prediction-box {
    display: flex;
    flex-direction: column;
    justify-content: flex-start; 
    align-items: center;
    padding-top: 0px;
    margin-top: 0px;
}

.instruction-text {
    text-align: center;
    font-size: 20px;
    font-weight: 600;
    color: #4b4b4b;
    margin-bottom: 20px;
}

@media (max-width: 768px) {
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: column-reverse !important;
    }
    .prediction-box {
        height: auto !important;
        margin-bottom: 20px;
    }
}
</style>
""", unsafe_allow_html=True)

if not st.session_state.show_canvas:
    st.markdown("<div class='draw-container'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='draw-text'>Click DRAW to make me predict what you draw (0 to 9)</div>",
        unsafe_allow_html=True
    )
    if st.button("DRAW"):
        st.session_state.show_canvas = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown('<div class="instruction-text">Try drawing any number between 0 - 9 below!</div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1])

    with col_left:
        canvas = st_canvas(
            fill_color="rgba(255,255,255,1)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key=st.session_state.canvas_key
        )

    with col_right:
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        if st.session_state.prediction is not None:
            if isinstance(st.session_state.prediction, (int, np.integer)):
                st.markdown(
                    f"""
                    <div style="text-align:center; color:#4b4b4b; line-height:1.2;">
                        <p style="font-size:24px; font-weight:700; margin:0; padding:0;">You drew:</p>
                        <span style="font-size:180px; font-weight:900; color:#2ecc71; display:block; margin:0; padding:0;">
                            {st.session_state.prediction}
                        </span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<h2 style='text-align:center;color:#f39c12; font-size:30px;'>{st.session_state.prediction}</h2>",
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.5, 1])
    with c2:
        if st.button("Predict"):
            if canvas.image_data is not None:
                img = canvas.image_data[:, :, :3].mean(axis=2).astype(np.uint8)
                digit_28 = Image.fromarray(img).resize((28, 28), resample=Image.BILINEAR)
                X = np.array(digit_28, dtype=np.uint8).reshape(1, -1)
                probs = model.predict_proba(X)[0]
                best_prob = probs.max()
                pred = probs.argmax()
                if best_prob < 0.5:
                    st.session_state.prediction = "Please redraw"
                else:
                    st.session_state.prediction = int(pred)
                st.rerun()

    if st.button("Clear"):
        st.session_state.canvas_key = f"canvas_{np.random.randint(1_000_000)}"
        st.session_state.prediction = None
        st.rerun()