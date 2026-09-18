"""
Pneumonia Detection — Streamlit App  (Capstone Project, Vignesh J)

Image classification: trained Keras CNN model (VGG16 Frozen)
Clinical context:    Groq LLM (same API used in the NewsFindr GenAI project)

Upload a chest X-ray (.dcm / .png / .jpg) → predicted class + probability + AI clinical context.
"""
import io, os, urllib.request
import numpy as np
import streamlit as st
import cv2
from PIL import Image
from tensorflow import keras

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

MODEL_PATH    = "pneumonia_best_model.keras"
MODEL_GDRIVE_ID = "17vcLLsZ8B7l4CB2EosE5mwDetFkBMc3i"   # Google Drive file ID
IMG_SIZE      = 128   # must match training IMG_SIZE

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="centered")


def download_model_if_needed():
    """Download model from Google Drive if not already present."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return  # already downloaded
    url = f"https://drive.google.com/uc?export=download&id={MODEL_GDRIVE_ID}"
    st.info("⬇️ Downloading model weights (~59 MB) on first run — please wait…")
    try:
        import gdown
        gdown.download(url, MODEL_PATH, quiet=False)
    except ImportError:
        # fallback: urllib (may fail for large files behind cookie wall)
        urllib.request.urlretrieve(url, MODEL_PATH)
    st.success("✅ Model downloaded.")


@st.cache_resource
def load_model():
    download_model_if_needed()
    return keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_groq_client():
    """Initialise Groq client from environment variable (set as a Space secret)."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if api_key and GROQ_AVAILABLE:
        return Groq(api_key=api_key)
    return None


def read_uploaded_image(uploaded_file) -> np.ndarray:
    """Return a single-channel (grayscale) uint8 numpy array from an uploaded
    DICOM, PNG, or JPG file."""
    raw = uploaded_file.read()
    if uploaded_file.name.lower().endswith(".dcm"):
        idx = raw.find(b"\xff\xd8\xff")
        if idx == -1:
            raise ValueError("Could not find embedded JPEG pixel data in this DICOM file.")
        img = Image.open(io.BytesIO(raw[idx:])).convert("L")
        return np.array(img)
    img = Image.open(io.BytesIO(raw)).convert("L")
    return np.array(img)


def preprocess(img_gray: np.ndarray, expects_rgb: bool) -> np.ndarray:
    resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    if expects_rgb:
        arr = cv2.cvtColor((normalized * 255).astype("uint8"), cv2.COLOR_GRAY2RGB).astype("float32") / 255.0
    else:
        arr = np.expand_dims(normalized, axis=-1)
    return np.expand_dims(arr, axis=0)


def get_groq_interpretation(predicted_class: str, probability: float, groq_client) -> str:
    """Call Groq LLM for a concise clinical context — same API as NewsFindr project."""
    prompt = (
        f"You are a radiology AI assistant. A CNN model analysed a chest X-ray and returned:\n"
        f"- Predicted class: {predicted_class}\n"
        f"- Probability of pneumonia: {probability:.1%}\n\n"
        "In 2-3 sentences, provide a concise clinical context: what this result means, "
        "what the radiologist should prioritise, and any key caveats. Do NOT make a definitive diagnosis."
    )
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.2,
    )
    return response.choices[0].message.content


# ── UI ──────────────────────────────────────────────────────────────────────
st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.write(
    "Upload a chest X-ray image (`.dcm`, `.png`, or `.jpg`) to get a predicted "
    "class (Pneumonia / No Pneumonia) and the model's confidence."
)
st.caption(
    "Decision-support demo for an academic capstone project — "
    "**not** a certified medical device and must not be used for real clinical diagnosis."
)

uploaded_file = st.file_uploader("Chest X-ray image", type=["dcm", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        model = load_model()
        groq_client = load_groq_client()
        expects_rgb = model.input_shape[-1] == 3

        img_gray = read_uploaded_image(uploaded_file)
        st.image(img_gray, caption="Uploaded X-ray", use_column_width=True, clamp=True)

        input_tensor = preprocess(img_gray, expects_rgb)
        probability = float(model.predict(input_tensor, verbose=0)[0, 0])
        predicted_class = "Pneumonia" if probability >= 0.5 else "No Pneumonia"

        st.subheader("Prediction")
        col1, col2 = st.columns(2)
        col1.metric("Predicted class", predicted_class)
        col2.metric("Probability of pneumonia", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))

        if predicted_class == "Pneumonia":
            st.warning("The model flags this image as **likely pneumonia-positive** — recommend prioritised radiologist review.")
        else:
            st.success("The model flags this image as **likely pneumonia-negative**.")

        # ── Groq LLM clinical context (same API as NewsFindr GenAI project) ──
        if groq_client:
            with st.spinner("Generating clinical context via Groq LLM…"):
                interpretation = get_groq_interpretation(predicted_class, probability, groq_client)
            if interpretation:
                st.subheader("Clinical Context (Groq LLM)")
                st.info(interpretation)
        else:
            st.caption(
                "💡 **Groq LLM not configured.** "
                "Add `GROQ_API_KEY` as a Streamlit secret to enable AI-powered clinical context."
            )

    except Exception as e:
        st.error(f"Could not process this file: {e}")
else:
    st.info("Upload an image above to run a prediction.")
