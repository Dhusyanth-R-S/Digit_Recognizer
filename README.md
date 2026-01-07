# ✍️ Handwritten Digit Recognizer (Classical Machine Learning)

This project is a handwritten digit recognition system built using **classical machine learning**, designed to work reliably with **real user drawings from a Streamlit canvas**.

The goal of this project is not just high notebook accuracy, but **stable, trustworthy behavior after deployment**.

---

# 🚩 Problem Statement

During initial development, the model performed well during training but struggled when deployed.

The core issue was **data mismatch**:
- Training data did not reflect how users actually draw digits
- Notebook accuracy did not translate to real-world performance
- Predictions were often low-confidence or unstable

This is a common challenge in applied machine learning systems.

---

# 🧠 Approach & Solution

Instead of switching to deep learning, the focus was placed on fixing the **data pipeline and real-world alignment**.

## Key steps taken

- Built a Streamlit-based digit recognition app
- Generated Streamlit-style synthetic data aligned with deployment input
- Created a separate Streamlit app to collect real handwritten digit samples
- Collected real user drawings in multiple styles
- Retrained the model using combined synthetic and real data
- Added a confidence-based safeguard for uncertain predictions

This approach significantly improved real-time reliability.

---

# ⚙️ Model Details

- Algorithm: Logistic Regression (LogisticRegressionCV)
- Type: Classical Machine Learning
- Input: 28×28 grayscale images (flattened to 784 features)
- Output: Digit prediction (0–9) with confidence score
- Inference: Real-time prediction via Streamlit
- UX Safeguard: Prompts the user to redraw when confidence is low

This keeps the system fast, interpretable, and lightweight.

---

# 📊 Results (Practical Evaluation)

Although notebook metrics improved significantly after retraining, the most important gains were observed in real usage:

- Fewer incorrect predictions
- More stable confidence scores
- Reduced redraw prompts
- Improved handling of messy or imperfect digits

### Checkout the app here 

[🚀 Live App Link](https://digitrecognizer-by-dhusya.streamlit.app/)

---

# 👤 Author

**Dhusyanth R S**

I am a data science learner with a strong interest in **classical machine learning, data analysis, and real-world problem solving**.  
I focus on building projects that go beyond notebook accuracy and instead work reliably in **practical, deployed environments**.

This project reflects my approach to machine learning:
- understanding data–model mismatch
- iterating through better data rather than blindly changing models
- designing simple, interpretable solutions
- validating performance through real usage, not just metrics

I enjoy working with Python, SQL, and ML libraries, and I am actively building projects that demonstrate end-to-end ownership — from data collection to deployment.

