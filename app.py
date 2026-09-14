import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from medical_qml_model import (
    IMG_SIZE, make_dataset, make_custom_scan, train_pipeline, predict_single,
    quantum_circuit, N_QLAYERS, N_QUBITS,
)
from pennylane import numpy as pnp
import pennylane as qml

st.set_page_config(page_title="Hybrid DL + QML — Medical Imaging", page_icon="🧬", layout="wide")

st.title("🧬 Hybrid Deep Learning + Quantum ML for Medical Imaging")
st.caption(
    "A small CNN feature extractor feeding a variational quantum classifier — "
    "inspired by Dr. Moulay Akhloufi's *Deep learning for medical imaging* project "
    "(Université de Moncton), scaled down to run in your browser in under a minute."
)
st.info(
    "⚠️ **This is an educational toy, not a medical device.** All images are "
    "computer-generated synthetic patterns, not real scans, and nothing here should "
    "inform any real diagnosis.",
    icon="⚠️",
)

# ---------------------------------------------------------------------------
# Sidebar: configuration + train button
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Pipeline settings")
n_per_class = st.sidebar.slider("Images per class", 20, 150, 60, step=10,
                                 help="Dataset size = 2 x this number. Bigger = slower, "
                                      "slightly more robust.")
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
cnn_epochs = st.sidebar.slider("CNN backbone epochs", 2, 16, 8)
q_iters = st.sidebar.slider("Quantum head iterations", 10, 80, 40, step=10)

st.sidebar.divider()
st.sidebar.caption(
    "Stage 1 (CNN) is pure NumPy with a hand-written backward pass, so it's the slow part "
    "(roughly proportional to images × epochs). Stage 2 (the 4-qubit circuit) is fast."
)

train_clicked = st.sidebar.button("🚀 Train the pipeline", type="primary", width='stretch')

if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.trained_config = None

current_config = (n_per_class, seed, cnn_epochs, q_iters)

@st.cache_resource(show_spinner=False)
def _run_training(n_per_class, seed, cnn_epochs, q_iters):
    progress_bar = st.progress(0.0, text="Starting...")

    def cb(frac, msg):
        progress_bar.progress(min(frac, 1.0), text=msg)

    result = train_pipeline(n_per_class=n_per_class, seed=seed, cnn_epochs=cnn_epochs,
                             q_iters=q_iters, progress_cb=cb)
    progress_bar.empty()
    return result

need_training = train_clicked or st.session_state.result is None
if need_training:
    with st.spinner("Training CNN backbone + quantum head... (roughly a few tens of seconds)"):
        st.session_state.result = _run_training(*current_config)
        st.session_state.trained_config = current_config

result = st.session_state.result
stale = st.session_state.trained_config != current_config
if stale:
    st.warning("Settings changed since the last training run — click **Train the pipeline** "
               "in the sidebar to retrain with the new settings.", icon="🔄")

tab_data, tab_train, tab_try = st.tabs(
    ["📊 Dataset", "🧠 Training & Results", "🎛️ Try it yourself"]
)

# ---------------------------------------------------------------------------
# Tab 1: Dataset
# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("Synthetic medical scans")
    st.write(
        "Each image is a 32×32 grayscale 'scan': soft tissue-like background texture, "
        "optionally with a bright, irregularly-shaped blob standing in for a tumor. "
        "No real medical dataset is used or needed."
    )
    preview_X, preview_y = make_dataset(n_per_class=4, size=IMG_SIZE, seed=int(seed))
    cols = st.columns(8)
    for i, col in enumerate(cols):
        with col:
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(preview_X[i], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            ax.set_title("tumor" if preview_y[i] == 1 else "normal", fontsize=9)
            st.pyplot(fig, width='stretch')
            plt.close(fig)

# ---------------------------------------------------------------------------
# Tab 2: Training & results
# ---------------------------------------------------------------------------
with tab_train:
    if result is None:
        st.write("Click **Train the pipeline** in the sidebar to get started.")
    else:
        st.subheader("Stage 1 — CNN backbone")
        st.write(
            "Conv2D(8, 3×3) → ReLU → MaxPool → Dense(16) → ReLU → **Dense(4) → Tanh** "
            "(bottleneck) → Dense(1) → Sigmoid. The final sigmoid head is discarded after "
            "training; only the 4-number bottleneck is kept."
        )
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(range(1, len(result["backbone_losses"]) + 1), result["backbone_losses"], marker="o")
            ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss"); ax.set_title("Backbone training loss")
            st.pyplot(fig, width='stretch')
            plt.close(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(range(1, len(result["backbone_accs"]) + 1),
                    [a * 100 for a in result["backbone_accs"]], marker="o", color="green")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105)
            ax.set_title("Backbone test accuracy")
            st.pyplot(fig, width='stretch')
            plt.close(fig)

        st.subheader("Learned features (frozen backbone output)")
        embed_test = result["embed_test"]
        y_test = result["y_test"]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for label, color, name in [(0, "tab:blue", "normal"), (1, "tab:red", "tumor")]:
            mask = y_test == label
            ax.scatter(embed_test[mask, 0], embed_test[mask, 1], c=color, label=name, alpha=0.6)
        ax.set_xlabel("learned feature 0"); ax.set_ylabel("learned feature 1")
        ax.legend()
        st.pyplot(fig, width='stretch')
        plt.close(fig)

        st.subheader("Stage 2 — quantum classifier head")
        st.write(
            f"{N_QUBITS} qubits, {N_QLAYERS} entangling layers. Angle-encodes the 4 CNN "
            "features, then a trainable variational circuit, measured as PauliZ on qubit 0."
        )
        example_weights = pnp.random.uniform(0, 2 * np.pi, size=(N_QLAYERS, N_QUBITS))
        example_x = (embed_test[0] + 1) / 2 * np.pi
        fig, ax = qml.draw_mpl(quantum_circuit, decimals=1, level="device")(example_weights, example_x)
        st.pyplot(fig, width='stretch')
        plt.close(fig)

        qh = result["q_history"]
        c3, c4 = st.columns(2)
        with c3:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(qh["iter"], qh["cost"], marker="o")
            ax.set_xlabel("Iteration"); ax.set_ylabel("Squared loss"); ax.set_title("Quantum head cost")
            st.pyplot(fig, width='stretch')
            plt.close(fig)
        with c4:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(qh["iter"], [a * 100 for a in qh["test_acc"]], marker="o", color="green")
            ax.set_xlabel("Iteration"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105)
            ax.set_title("Quantum head test accuracy")
            st.pyplot(fig, width='stretch')
            plt.close(fig)

        st.subheader("Final evaluation (full pipeline, held-out test set)")
        conf = result["confusion"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("True positives", conf["tp"])
        m2.metric("False positives", conf["fp"])
        m3.metric("True negatives", conf["tn"])
        m4.metric("False negatives", conf["fn"])
        denom_p = conf["tp"] + conf["fp"]
        denom_r = conf["tp"] + conf["fn"]
        precision = conf["tp"] / denom_p if denom_p else 0.0
        recall = conf["tp"] / denom_r if denom_r else 0.0
        st.write(f"**Precision:** {precision:.0%}  •  **Recall:** {recall:.0%}")

        st.subheader("Sample predictions")
        X_test_img = result["X_test_img"]
        preds_test = result["preds_test"]
        n_show = min(8, len(X_test_img))
        cols = st.columns(n_show)
        for i, col in enumerate(cols):
            with col:
                true_l = "tumor" if y_test[i] == 1 else "normal"
                pred_l = "tumor" if preds_test[i] == 1 else "normal"
                ok = preds_test[i] == (2 * y_test[i] - 1)
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.imshow(X_test_img[i], cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                ax.set_title(f"true:{true_l}\npred:{pred_l}",
                             color="green" if ok else "red", fontsize=8)
                st.pyplot(fig, width='stretch')
                plt.close(fig)

# ---------------------------------------------------------------------------
# Tab 3: interactive playground
# ---------------------------------------------------------------------------
with tab_try:
    st.subheader("Build your own scan and see the model's live prediction")
    if result is None:
        st.write("Train the pipeline first (sidebar) to use this tab.")
    else:
        left, right = st.columns([1, 1])
        with left:
            has_tumor = st.checkbox("Include a mass", value=True)
            bx = st.slider("Mass position — x", 0.1, 0.9, 0.5, disabled=not has_tumor)
            by = st.slider("Mass position — y", 0.1, 0.9, 0.5, disabled=not has_tumor)
            radius = st.slider("Mass radius", 2.0, 9.0, 5.0, disabled=not has_tumor)
            intensity = st.slider("Mass brightness", 0.15, 0.6, 0.45, disabled=not has_tumor)
            noise = st.slider("Background noise", 0.0, 0.08, 0.03)
            img_seed = st.number_input("Background seed", 0, 9999, 7)

            img = make_custom_scan(has_tumor, bx, by, radius, intensity, noise, seed=int(img_seed))
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            st.pyplot(fig, width='stretch')
            plt.close(fig)

        with right:
            pred = predict_single(result["backbone"], result["q_weights"], result["q_bias"], img)
            if pred["label"] == "tumor":
                st.error(f"### Prediction: 🔴 tumor  \nConfidence: {pred['confidence']:.0%}")
            else:
                st.success(f"### Prediction: 🟢 normal  \nConfidence: {pred['confidence']:.0%}")

            st.write("**4 learned features fed to the quantum circuit** (CNN bottleneck, tanh-bounded):")
            fig, ax = plt.subplots(figsize=(4.5, 2.2))
            ax.bar(range(4), pred["bottleneck"], color="tab:purple")
            ax.set_xticks(range(4))
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set_xlabel("bottleneck dimension")
            st.pyplot(fig, width='stretch')
            plt.close(fig)

            st.caption(f"Raw quantum circuit output (PauliZ expectation + bias): "
                       f"`{pred['raw_score']:.3f}` — positive leans tumor, negative leans normal.")

st.divider()
st.caption(
    "Reference: inspired by Dr. Moulay Akhloufi's *Deep learning for medical imaging* project "
    "(Université de Moncton), which lists Quantum ML alongside deep learning as a research "
    "direction. Not affiliated with or reviewed by that lab."
)
