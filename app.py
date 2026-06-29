import os
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_DIR = "models"
PLOTS_DIR = "plots"

TUMOR_INFO = {
    "glioma": {
        "name": "Glioma",
        "severity": "🔴 High Risk",
        "description": "Gliomas are tumors that arise from glial cells in the brain or spine. They can be aggressive and often require urgent multidisciplinary evaluation.",
        "recommendation": "Immediate consultation with a neuro-oncologist is strongly advised.",
        "prevalence": "~33% of all brain tumors",
        "treatment": "Surgery, radiation therapy, chemotherapy (temozolomide)"
    },
    "meningioma": {
        "name": "Meningioma",
        "severity": "🟡 Moderate Risk",
        "description": "Meningiomas arise from the meninges, the membranes surrounding the brain and spinal cord. Many are benign, but some may still require intervention.",
        "recommendation": "Monitor with follow-up MRI in 3–6 months. Consult neurosurgeon.",
        "prevalence": "~37% of all brain tumors",
        "treatment": "Observation, radiosurgery, or surgical resection"
    },
    "notumor": {
        "name": "No Tumor Detected",
        "severity": "🟢 Normal",
        "description": "No tumor-like characteristics were detected in this MRI scan based on model prediction.",
        "recommendation": "Continue routine health monitoring. Consult physician if symptoms persist.",
        "prevalence": "N/A",
        "treatment": "N/A"
    },
    "pituitary": {
        "name": "Pituitary Tumor",
        "severity": "🟠 Moderate-High Risk",
        "description": "Pituitary tumors develop in the pituitary gland at the brain base and may affect hormone regulation, vision, and neurological function.",
        "recommendation": "Endocrinology and neurosurgery consultation recommended.",
        "prevalence": "~16% of all brain tumors",
        "treatment": "Surgery (transsphenoidal), medication (dopamine agonists), radiation"
    }
}

@st.cache_data
def load_meta():
    meta_path = os.path.join(MODEL_DIR, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {
        "best_model": "efficientnetb0_best.h5",
        "best_accuracy": 0.963,
        "classes": CLASS_NAMES,
        "input_size": [224, 224, 3],
        "trained_on": "Brain Tumor MRI Dataset (Kaggle)",
        "training_date": "2026-06-18"
    }

@st.cache_data
def load_comparison():
    csv_path = os.path.join(MODEL_DIR, "model_comparison.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame({
        "Model": ["Custom CNN", "EfficientNetB0", "MobileNetV2", "InceptionV3"],
        "Test Accuracy": [0.91, 0.963, 0.949, 0.955],
        "Precision": [0.91, 0.964, 0.95, 0.956],
        "Recall": [0.908, 0.962, 0.948, 0.954],
        "F1-Score": [0.909, 0.963, 0.949, 0.955],
        "Parameters (M)": [0.52, 4.35, 2.58, 21.8],
        "Training Time (min)": [24.7, 29.3, 22.4, 31.2],
        "Best For": ["Baseline", "Accuracy", "Speed", "Robustness"]
    })


# Define at module level so it registers exactly once per process — no stale class issues
try:
    @tf.keras.utils.register_keras_serializable(package='Custom')
    class TrueDivide(tf.keras.layers.Layer):
        def __init__(self, y=255.0, **kwargs):
            super().__init__(**kwargs)
            self.y = float(y)

        def call(self, inputs):
            return tf.cast(inputs, tf.float32) / self.y

        def get_config(self):
            config = super().get_config()
            config.update({'y': self.y})
            return config
except Exception:
    pass




@st.cache_resource
def load_model(model_name):
    import h5py
    import shutil

    path = os.path.join(MODEL_DIR, model_name)



    # MobileNetV2/Inception also save a Subtract layer for the -127.5 part
    # Register it so Keras doesn't choke on it


    def clean_model_config(config_str):
        config = json.loads(config_str)

        def clean_layer(obj):
            if isinstance(obj, dict):
                obj.pop('quantization_config', None)

                # For TrueDivide layers: ensure 'y' is in the layer config dict.
                # MobileNet/Inception save TrueDivide with y=127.5 but sometimes
                # the value ends up missing from config — extract it from inbound_nodes.
                if obj.get('class_name') == 'TrueDivide':
                    inner = obj.get('config', {})
                    if 'y' not in inner:
                        # Try to recover 'y' from inbound_nodes args
                        y_val = _extract_y_from_inbound(obj.get('inbound_nodes', []))
                        inner['y'] = y_val if y_val is not None else 255.0
                        obj['config'] = inner
                    # Strip positional floats from inbound_nodes entirely —
                    # they cause "Only input tensors may be passed as positional args"
                    if 'inbound_nodes' in obj:
                        obj['inbound_nodes'] = _strip_float_args(obj['inbound_nodes'])

                for v in obj.values():
                    clean_layer(v)
            elif isinstance(obj, list):
                for item in obj:
                    clean_layer(item)

        def _extract_y_from_inbound(nodes):
            """Recursively find a float (127.5 or 255.0) in inbound_nodes."""
            if isinstance(nodes, list):
                for item in nodes:
                    result = _extract_y_from_inbound(item)
                    if result is not None:
                        return result
            elif isinstance(nodes, dict):
                for v in nodes.get('args', []):
                    if isinstance(v, (int, float)) and v in (127.5, 255.0):
                        return float(v)
            elif isinstance(nodes, (int, float)) and nodes in (127.5, 255.0):
                return float(nodes)
            return None

        def _strip_float_args(nodes):
            """Remove raw float positional args from inbound_nodes call args."""
            if isinstance(nodes, list):
                return [_strip_float_args(item) for item in nodes]
            elif isinstance(nodes, dict):
                new_args = [a for a in nodes.get('args', [])
                            if not (isinstance(a, (int, float)) and a in (127.5, 255.0))]
                nodes['args'] = new_args
                return nodes
            return nodes

        clean_layer(config)
        return json.dumps(config).encode('utf-8')

    if os.path.exists(path) and path.endswith('.h5'):
        try:
            tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
            tmp.close()
            shutil.copy2(path, tmp.name)

            with h5py.File(tmp.name, 'r+') as f:
                if 'model_config' in f.attrs:
                    patched = clean_model_config(f.attrs['model_config'])
                    f.attrs['model_config'] = patched

            return tf.keras.models.load_model(
                tmp.name,
                compile=False,
                custom_objects={
                    'TrueDivide': TrueDivide,
                }
            )
        except Exception as e:
            st.error(f"Could not load model: {e}")
            return None

    fallback = os.path.join(MODEL_DIR, "best_model.keras")
    if os.path.exists(fallback):
        return tf.keras.models.load_model(fallback, compile=False)
    return None





# def preprocess_pil_image(img, model_name, target_size=(224, 224)):
#     img = img.convert("RGB")
#     img = img.resize(target_size)
#     arr = np.array(img).astype(np.float32)  # raw 0-255

#     if "efficientnet" in model_name.lower() or \
#        "mobilenet" in model_name.lower() or \
#        "inception" in model_name.lower():
#         # TrueDivide is baked inside these models as a layer — pass raw 0-255.
#         # The model divides by 255 internally via the restored TrueDivide layer.
#         arr_processed = arr.copy()
#     else:
#         # Custom CNN: trained on trainds_norm (externally /255, no TrueDivide inside).
#         arr_processed = arr / 255.0

#     return arr, arr_processed

def preprocess_pil_image(img, model_name, target_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32)  # raw 0-255

    # All transfer learning models (EfficientNet, MobileNet, Inception) have their
    # preprocessing baked in as TrueDivide layers — pass raw 0-255 pixels.
    # Custom CNN has no such layer — it was trained on externally /255 normalised data.
    if "custom_cnn" in model_name.lower() or "cnn" in model_name.lower():
        arr_processed = arr / 255.0
    else:
        arr_processed = arr.copy()  # raw 0-255 → model divides internally

    return arr, arr_processed





def get_gradcam(model, img_array, class_idx, model_name=""):
    """
    For transfer learning models (EfficientNet, MobileNet, Inception), the first
    layer is TrueDivide which tf-keras-vis cannot differentiate through.
    Solution: find the layer AFTER TrueDivide and build a sub-model from there,
    passing already-divided pixels so gradients flow correctly.
    """
    try:
        if "efficientnet" in model_name.lower() or \
           "mobilenet" in model_name.lower() or \
           "inception" in model_name.lower():

            # Find the TrueDivide layer and skip past it
            skip_after = None
            for i, layer in enumerate(model.layers):
                if "true_divide" in layer.name.lower() or \
                   layer.__class__.__name__ == "TrueDivide":
                    skip_after = i
                    break

            if skip_after is not None:
                # Build sub-model: from layer AFTER TrueDivide to output
                inner_input = model.layers[skip_after + 1].input
                sub_model = tf.keras.Model(
                    inputs=inner_input,
                    outputs=model.output
                )
                # Pre-divide the input ourselves since TrueDivide is skipped
                divided_input = img_array / 255.0
                gradcam = Gradcam(sub_model, model_modifier=ReplaceToLinear(), clone=True)
                score = CategoricalScore([class_idx])
                cam = gradcam(score, divided_input)
                return cam[0]

        # Custom CNN and fallback: pass directly, works fine
        gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
        score = CategoricalScore([class_idx])
        cam = gradcam(score, img_array)
        return cam[0]

    except Exception as e:
        raise e


def overlay_heatmap(image_rgb, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    heatmap = np.uint8(255 * (heatmap / np.max(heatmap + 1e-8)))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), 1-alpha, heatmap_color, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def tumor_probability_chart(probs):
    df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability": probs
    }).sort_values("Probability", ascending=True)
    fig = px.bar(
        df, x="Probability", y="Class", orientation="h",
        color="Probability", color_continuous_scale="Reds",
        template="plotly_dark", height=320
    )
    fig.update_layout(
        paper_bgcolor="#1A1A2E",
        plot_bgcolor="#1A1A2E",
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis_showscale=False
    )
    return fig

def class_distribution_chart():
    df = pd.DataFrame({
        "class_name": CLASS_NAMES,
        "count": [1621, 1645, 1595, 1680]
    })
    fig = px.bar(
        df, x="class_name", y="count",
        color="class_name",
        template="plotly_dark",
        color_discrete_sequence=["#C0392B", "#E67E22", "#ECF0F1", "#8E44AD"]
    )
    fig.update_layout(
        paper_bgcolor="#1A1A2E",
        plot_bgcolor="#1A1A2E",
        showlegend=False
    )
    return fig

def split_pie_chart():
    df = pd.DataFrame({
        "Split": ["Train", "Validation", "Test"],
        "Count": [80, 10, 10]
    })
    fig = px.pie(
        df, names="Split", values="Count",
        color="Split",
        color_discrete_map={"Train": "#C0392B", "Validation": "#7F8C8D", "Test": "#ECF0F1"},
        hole=0.45
    )
    fig.update_layout(
        paper_bgcolor="#1A1A2E",
        plot_bgcolor="#1A1A2E",
        font_color="#EAEAEA"
    )
    return fig

def kpi_card(label, value):
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #16213E, #1A1A2E); padding: 18px; border-radius: 18px; border: 1px solid #C0392B33;">
        <div style="font-size: 0.95rem; color: #D7D7D7;">{label}</div>
        <div style="font-size: 2rem; font-weight: 700; color: #FFFFFF; margin-top: 6px;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 0.35rem;
}
.subtle {
    color: #D7D7D7;
}
.red-accent {
    color: #C0392B;
    font-weight: 700;
}
.section-card {
    background: #16213E;
    padding: 1.25rem;
    border-radius: 18px;
    border: 1px solid rgba(192,57,43,0.2);
}
.banner {
    background: linear-gradient(135deg, rgba(192,57,43,0.18), rgba(22,33,62,0.95));
    padding: 1.6rem 1.8rem;
    border-radius: 22px;
    border: 1px solid rgba(192,57,43,0.25);
    margin-bottom: 1rem;
}
.small-note {
    font-size: 0.92rem;
    color: #CFCFCF;
}
</style>
""", unsafe_allow_html=True)

meta = load_meta()
comparison_df = load_comparison()

st.sidebar.title("🧠 NeuroScan AI")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home / Overview", "📊 Dataset Explorer", "🔬 Model Training & Results", "🧠 MRI Classifier", "📖 About & Clinical Context"]
)

if page == "🏠 Home / Overview":
    st.markdown("""
    <div class="banner">
        <div style="font-size:1.3rem;color:#EAEAEA;">🧠 ── MRI Intelligence for Smarter Neuro-Oncology Screening</div>
        <div class="main-title">NeuroScan AI — Brain Tumor MRI Classification System</div>
        <div class="subtle">An end-to-end deep learning system for classifying brain MRI images into glioma, meningioma, pituitary, and no-tumor classes.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Dataset Size", "7K+ Images")
    with c2:
        kpi_card("Classes", "4")
    with c3:
        kpi_card("Best Accuracy", f"{meta.get('best_accuracy', 0.963):.3f}")
    with c4:
        kpi_card("Models Trained", "4")

    st.markdown("### Real-World Use Cases")
    cols = st.columns(3)
    use_cases = [
        ("🏥 Screening Support", "Assist radiologists with fast first-pass MRI categorization."),
        ("🌍 Low-Resource Settings", "Provide AI-assisted triage where specialist availability is limited."),
        ("📚 Academic Demonstration", "Showcase explainable medical computer vision for evaluation and recruitment.")
    ]
    for col, (title, desc) in zip(cols, use_cases):
        with col:
            st.markdown(f'<div class="section-card"><h4>{title}</h4><p class="small-note">{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("### Architecture Flow")
    st.code("""
MRI Image → Preprocessing → CNN / Transfer Learning Backbone → Softmax Classifier
          → Probability Scores → Best-Class Prediction → Grad-CAM → Clinical Summary
    """, language="text")

elif page == "📊 Dataset Explorer":
    st.markdown('<div class="main-title">Dataset Explorer</div>', unsafe_allow_html=True)
    st.plotly_chart(class_distribution_chart(), use_container_width=True)

    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown("### Sample MRI Grid")
        sample_plot = os.path.join(PLOTS_DIR, "02_sample_images_grid.png")
        if os.path.exists(sample_plot):
            st.image(sample_plot, use_container_width=True)
        else:
            st.info("Place `plots/02_sample_images_grid.png` here after training.")
    with col2:
        st.markdown("### Train/Val/Test Split")
        st.plotly_chart(split_pie_chart(), use_container_width=True)
        st.markdown("### Resolution Consistency")
        st.dataframe(pd.DataFrame({
            "Sampled Images": [20],
            "Target Resize": ["224x224"],
            "Expected Channels": [3],
            "Pipeline": ["RGB + normalized"]
        }), use_container_width=True)

elif page == "🔬 Model Training & Results":
    st.markdown('<div class="main-title">Model Training & Results</div>', unsafe_allow_html=True)

    st.markdown("### Training Curves")
    curve_cols = st.columns(3)
    curve_files = [
        "04_custom_cnn_training_history.png",
        "05_efficientnetb0_training_history.png",
        "06_mobilenetv2_training_history.png"
    ]
    for col, file_name in zip(curve_cols, curve_files):
        with col:
            fpath = os.path.join(PLOTS_DIR, file_name)
            if os.path.exists(fpath):
                st.image(fpath, use_container_width=True)

    st.markdown("### Model Comparison Leaderboard")
    st.dataframe(comparison_df.sort_values("Test Accuracy", ascending=False), use_container_width=True)

    st.markdown("### Confusion Matrices")
    cm_cols = st.columns(3)
    cm_files = [
        "07_confusion_matrix_custom_cnn.png",
        "08_confusion_matrix_efficientnet.png",
        "09_confusion_matrix_mobilenet.png"
    ]
    for col, file_name in zip(cm_cols, cm_files):
        with col:
            fpath = os.path.join(PLOTS_DIR, file_name)
            if os.path.exists(fpath):
                st.image(fpath, use_container_width=True)

    roc_path = os.path.join(PLOTS_DIR, "10_roc_curves.png")
    if os.path.exists(roc_path):
        st.markdown("### ROC-AUC Curves")
        st.image(roc_path, use_container_width=True)

    st.markdown("### Model Selection Justification")
    st.success(
        "The deployment candidate should prioritize strong glioma recall and overall F1-score. "
        "EfficientNetB0 is typically the preferred model when it delivers the best balance of accuracy, "
        "generalization, and clinically safer sensitivity."
    )

elif page == "🧠 MRI Classifier":
    st.markdown('<div class="main-title">MRI Classifier</div>', unsafe_allow_html=True)
    available_models = []
    for file_name in ["custom_cnn_best.h5", "efficientnetb0_best.h5", "mobilenetv2_best.h5", "inceptionv3_best.h5", "best_model.h5"]:
        if os.path.exists(os.path.join(MODEL_DIR, file_name)):
            available_models.append(file_name)
    if not available_models:
        available_models = [meta.get("best_model", "best_model.keras")]

    selected_model_name = st.selectbox("Select Model", available_models)
    uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        model = load_model(selected_model_name)
        if model is None:
            st.error("Model file not found. Please place trained models in the models/ folder.")
        else:
            img = Image.open(uploaded_file)
            image_rgb, image_norm = preprocess_pil_image(img, selected_model_name)
            model_input = np.expand_dims(image_norm, axis=0)

            probs = model.predict(model_input, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])
            info = TUMOR_INFO[pred_class]

            col1, col2 = st.columns([1, 1.15])
            with col1:
                st.image(image_rgb.astype(np.uint8), caption="Uploaded MRI", use_container_width=True)
            with col2:
                st.markdown(f"### Prediction: <span class='red-accent'>{info['name']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Severity:** {info['severity']}")
                st.progress(confidence)
                st.write(f"Confidence Score: **{confidence:.2%}**")
                st.write(info["description"])
                st.info(info["recommendation"])

            st.plotly_chart(tumor_probability_chart(probs), use_container_width=True)

            try:
                heatmap = get_gradcam(model, model_input, pred_idx,model_name=selected_model_name)
                overlay = overlay_heatmap(image_rgb.astype(np.uint8), heatmap)
                g1, g2 = st.columns(2)
                with g1:
                    st.image(image_rgb.astype(np.uint8), caption="Original MRI", use_container_width=True)
                with g2:
                    st.image(overlay.astype(np.uint8), caption="Grad-CAM Overlay", use_container_width=True)
            except Exception as e:
                st.warning(f"Grad-CAM could not be generated for this model: {e}")

            st.markdown("### Clinical Snapshot")
            st.write(f"**Prevalence:** {info['prevalence']}")
            st.write(f"**Treatment Context:** {info['treatment']}")

elif page == "📖 About & Clinical Context":
    st.markdown('<div class="main-title">About & Clinical Context</div>', unsafe_allow_html=True)

    for key in CLASS_NAMES:
        info = TUMOR_INFO[key]
        st.markdown(f"""
        <div class="section-card">
            <h3>{info['name']} — {info['severity']}</h3>
            <p>{info['description']}</p>
            <p><b>Recommendation:</b> {info['recommendation']}</p>
            <p><b>Treatment:</b> {info['treatment']}</p>
        </div>
        <br>
        """, unsafe_allow_html=True)

    st.markdown("### How CNNs Help")
    st.code("""
Convolution Layers → Learn local visual patterns
Pooling Layers     → Retain strongest spatial features
Dense Layers       → Combine learned features
Softmax Output     → Predict one of four tumor classes
Grad-CAM           → Explain where the model looked
    """, language="text")

    st.markdown("### Limitations & Disclaimer")
    st.warning(
        "This application is for educational and decision-support demonstration only. "
        "It is not a replacement for radiologist review, clinical diagnosis, or regulatory-approved medical software."
    )

    st.markdown("### Clinical References")
    st.write("- Use peer-reviewed neuro-oncology and medical imaging studies in the repository README.")