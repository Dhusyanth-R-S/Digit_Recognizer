import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import streamlit.components.v1 as components

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
if "draw_clicked" not in st.session_state:
    st.session_state.draw_clicked = False

if not st.session_state.show_canvas:
    st.markdown(
        "<h3 style='text-align:center;'>Click draw to make me recognize what you draw!</h3>",
        unsafe_allow_html=True
    )

    clicked = components.html(
        """
        <div style="display:flex;justify-content:center;">
            <button onclick="parent.postMessage('draw', '*')"
            style="
                width:200px;
                height:200px;
                border-radius:50%;
                font-size:32px;
                font-weight:800;
                border:none;
                cursor:pointer;
            ">
                DRAW
            </button>
        </div>
        <script>
        window.addEventListener("message", (e) => {
            if (e.data === "draw") {
                const streamlitDoc = window.parent.document;
                const buttons = streamlitDoc.querySelectorAll("button[kind='secondary']");
                buttons.forEach(b => {
                    if (b.innerText === "HIDDEN_DRAW_TRIGGER") {
                        b.click();
                    }
                });
            }
        });
        </script>
        """,
        height=240
    )

    if st.button("HIDDEN_DRAW_TRIGGER"):
        st.session_state.show_canvas = True
        st.rerun()

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
