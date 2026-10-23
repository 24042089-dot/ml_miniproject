
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import tempfile
import os

from pathlib import Path

st.set_page_config(
    page_title="Acoustic Fault Detection",
    page_icon="🎧",
    layout="wide"
)

st.title("🎧 Acoustic Anomaly Detection")
st.write(
    "ML-based detection of abnormal fan sounds using "
    "acoustic features and Random Forest."
)

MODEL_PATH = Path(__file__).parent / "random_forest_model.joblib"

@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["threshold"], saved["feature_cols"]

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    if len(y) < 1024:
        raise ValueError("Audio is too short. Upload a longer WAV recording.")

    features = []

    def add_stats(matrix):
        matrix = np.asarray(matrix)
        features.extend(np.mean(matrix, axis=1).tolist())
        features.extend(np.std(matrix, axis=1).tolist())

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=512
    )
    add_stats(mfcc)

    delta = librosa.feature.delta(mfcc)
    add_stats(delta)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=20
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    add_stats(mel_db)

    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=1024, hop_length=512
    )
    add_stats(contrast)

    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=1024, hop_length=512
    )
    add_stats(chroma)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    for feature in [centroid, bandwidth, rolloff, zcr, rms]:
        features.extend([
            float(np.mean(feature)),
            float(np.std(feature))
        ])

    return np.asarray(features, dtype=np.float64), y, sr

try:
    model, threshold, feature_cols = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a fan audio recording (.wav)",
    type=["wav"]
)

if uploaded_file is not None:
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav"
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        with st.spinner("Analysing audio..."):
            features, y, sr = extract_features(temp_path)

            if len(features) != len(feature_cols):
                st.error(
                    f"Feature mismatch: audio generated {len(features)} "
                    f"features, but model expects {len(feature_cols)}. "
                    "The training and app feature extraction must match."
                )
                st.stop()

            X = pd.DataFrame([features], columns=feature_cols)
            probability = float(model.predict_proba(X)[0, 1])
            is_anomaly = probability >= threshold

        st.success("Audio analysis complete!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction")
            if is_anomaly:
                st.error("⚠️ Anomalous sound detected")
            else:
                st.success("✅ Normal sound")

            st.metric("Anomaly score", f"{probability:.1%}")
            st.caption(
                "This score is the model's estimated anomaly-class "
                "probability, not a guaranteed measure of fault severity."
            )

        with col2:
            st.subheader("Audio details")
            st.write(f"**Sample rate:** {sr} Hz")
            st.write(f"**Duration:** {len(y) / sr:.2f} seconds")
            st.write(f"**Features extracted:** {len(features)}")

        st.subheader("Waveform")
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Spectrogram")
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=1024, hop_length=512)),
            ref=np.max
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(
            D, sr=sr, hop_length=512,
            x_axis="time", y_axis="hz", ax=ax
        )
        fig.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
        ax.set_title("Audio Spectrogram")
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Could not analyse this audio: {e}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
