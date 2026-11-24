"""
GenAI Day - AI vs Human Text Detection Frontend
Streamlit app connecting all 9 combinations of NLP feature pipelines and ML classifiers.
"""
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Ensure offline huggingface cache is respected without network delays
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import app_features

# ----------------------------------------------------------------------------
# CONFIG & CONSTANTS
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ART_DIR = BASE_DIR / "artifacts"

FEATURE_SETS = {
    "tfidf": {
        "title": "TF-IDF N-grams",
        "icon": ":material/format_list_bulleted:",
        "desc": "Sparse lexical vocabulary (20,000 n-gram features)",
    },
    "embeddings": {
        "title": "Dense Embeddings",
        "icon": ":material/hub:",
        "desc": "Semantic contextual vectors (384-dim MiniLM-L6-v2)",
    },
    "linguistic": {
        "title": "Stylometric & Linguistic",
        "icon": ":material/spellcheck:",
        "desc": "10 syntactic, lexical diversity & cadence features",
    },
}

CLASSIFIERS = {
    "logreg": {
        "title": "Logistic Regression",
        "icon": ":material/tune:",
        "desc": "Linear probabilistic model",
    },
    "rf": {
        "title": "Random Forest",
        "icon": ":material/forest:",
        "desc": "Ensemble of bagging decision trees",
    },
    "svm": {
        "title": "Linear SVM",
        "icon": ":material/timeline:",
        "desc": "Support vector machine with calibrated probabilities",
    },
}

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LABELS = {"0": "Human-Written", "1": "AI-Generated"}

# Page configuration
st.set_page_config(
    page_title="GenAI Day - 9-Model AI Text Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# CACHED ARTIFACTS AND EMBEDDER
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained models and vectorizers...")
def load_all_artifacts():
    """Loads all 9 models and auxiliary extractors."""
    models: Dict[Tuple[str, str], Any] = {}
    extras: Dict[str, Any] = {}
    missing: List[str] = []

    for feat in FEATURE_SETS:
        for clf in CLASSIFIERS:
            path = ART_DIR / f"{feat}__{clf}.joblib"
            if path.exists():
                models[(feat, clf)] = joblib.load(path)
            else:
                missing.append(path.name)

    for aux_name in ["tfidf_vectorizer", "linguistic_scaler"]:
        aux_path = ART_DIR / f"{aux_name}.joblib"
        if aux_path.exists():
            extras[aux_name] = joblib.load(aux_path)
        else:
            missing.append(aux_path.name)

    return models, extras, missing


@st.cache_resource(show_spinner="Loading sentence embedding transformer...")
def get_sentence_embedder(model_name: str):
    """Loads SentenceTransformer with fallback to offline local files."""
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


# ----------------------------------------------------------------------------
# FEATURE EXTRACTION & INFERENCE
# ----------------------------------------------------------------------------
def extract_feature_vector(feat_type: str, text: str, extras: dict, embed_name: str) -> np.ndarray:
    """Extracts features for the specified pipeline."""
    if feat_type == "tfidf":
        return extras["tfidf_vectorizer"].transform([text])
    elif feat_type == "embeddings":
        embedder = get_sentence_embedder(embed_name)
        return embedder.encode([text])
    elif feat_type == "linguistic":
        return app_features.linguistic_vector(text, extras["linguistic_scaler"])
    else:
        raise ValueError(f"Unknown feature pipeline: {feat_type}")


def predict_single_model(model, X, label_map: dict) -> Tuple[str, float, Dict[str, float]]:
    """Generates prediction label, confidence score, and probability distribution."""
    pred = model.predict(X)[0]
    classes = list(getattr(model, "classes_", [0, 1]))

    # Obtain calibrated probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
    elif hasattr(model, "decision_function"):
        dec = np.atleast_1d(model.decision_function(X)[0])
        if dec.size == 1:
            p1 = 1 / (1 + np.exp(-dec[0]))
            probs = np.array([1 - p1, p1])
        else:
            exp_d = np.exp(dec - dec.max())
            probs = exp_d / exp_d.sum()
    else:
        probs = np.ones(len(classes)) / len(classes)

    pred_idx = classes.index(pred) if pred in classes else 0
    conf = float(probs[pred_idx])

    prob_dict = {
        label_map.get(str(cls_val), str(cls_val)): float(p)
        for cls_val, p in zip(classes, probs)
    }

    display_label = label_map.get(str(pred), str(pred))
    return display_label, conf, prob_dict


# ----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------------------------------
models, extras, missing_files = load_all_artifacts()

with st.sidebar:
    st.markdown("### ⚙️ Model configuration")
    st.caption("Connect and orchestrate the 9 ML & NLP model combinations")

    if missing_files:
        st.error(f"Missing {len(missing_files)} artifact files:\n- " + "\n- ".join(missing_files), icon=":material/warning:")
    else:
        st.success(f"All 9 models + 2 vectorizers loaded", icon=":material/check_circle:")

    with st.expander("Pipeline selection", expanded=True, icon=":material/tune:"):
        sel_all = st.checkbox("Select all 9 combinations (Ensemble)", value=True)

        if sel_all:
            sel_feats = list(FEATURE_SETS.keys())
            sel_clfs = list(CLASSIFIERS.keys())
            st.info("Using complete 9-model ensemble: TF-IDF, Embeddings, & Linguistic with LogReg, Random Forest, & SVM.")
        else:
            sel_feats = st.multiselect(
                "Feature pipelines",
                options=list(FEATURE_SETS.keys()),
                default=list(FEATURE_SETS.keys()),
                format_func=lambda k: FEATURE_SETS[k]["title"],
            )
            sel_clfs = st.multiselect(
                "Classifiers",
                options=list(CLASSIFIERS.keys()),
                default=list(CLASSIFIERS.keys()),
                format_func=lambda k: CLASSIFIERS[k]["title"],
            )

    with st.expander("Model & OCR settings", expanded=False, icon=":material/settings:"):
        embed_name = st.text_input(
            "Embedding model",
            value=DEFAULT_EMBED_MODEL,
            help="HuggingFace sentence-transformer checkpoint for dense representations",
        )
        tess_default = app_features.get_default_tesseract_path() or ""
        tess_path_input = st.text_input(
            "Tesseract OCR path (optional)",
            value=tess_default,
            placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            help="Path to tesseract binary. Automatically populated if installed in standard path.",
        )
        ocr_engine = st.segmented_control(
            "OCR engine preference",
            options=["auto", "tesseract", "easyocr"],
            default="auto",
            format_func=lambda x: {"auto": "Auto-detect", "tesseract": "Tesseract", "easyocr": "EasyOCR"}[x],
        )

    with st.expander("Label mapping", expanded=False, icon=":material/label:"):
        label_human = st.text_input("Label for Class 0", value="Human-Written")
        label_ai = st.text_input("Label for Class 1", value="AI-Generated")
        label_map = {"0": label_human, "1": label_ai}

    st.divider()
    st.markdown("**Project details**")
    st.caption("GenAI Day Multi-Model Benchmark Suite\nArtifacts: `genai_day/artifacts`")


# ----------------------------------------------------------------------------
# HEADER & MAIN INTERFACE
# ----------------------------------------------------------------------------
st.title("🧠 GenAI Day - AI vs Human Text Detection")
st.markdown(
    "A multimodal text classification platform connecting **9 distinct model combinations** "
    "(3 NLP feature representations $\\times$ 3 ML classifiers) with ensembled consensus, "
    "stylometric inspection, and benchmark comparisons."
)

tab_analyze, tab_benchmarks, tab_architecture = st.tabs([
    "🔍 Analyze input",
    "📊 Benchmarks & performance",
    "📐 Model architecture",
])


# ============================================================================
# TAB 1: ANALYZE INPUT (TEXT / PDF / IMAGE)
# ============================================================================
with tab_analyze:
    st.subheader("1. Input selection", anchor=False)

    input_mode = st.segmented_control(
        "Choose input type",
        options=["Text input", "Upload PDF", "Upload image", "Sample presets"],
        default="Text input",
    )

    current_text = ""
    input_metadata = {}

    if input_mode == "Sample presets":
        st.markdown("Select a pre-loaded sample text to test the 9-model ensemble immediately:")
        chosen_sample = st.segmented_control(
            "Choose a sample",
            options=list(app_features.SAMPLE_TEXTS.keys()),
            default=list(app_features.SAMPLE_TEXTS.keys())[0],
        )
        if chosen_sample:
            current_text = app_features.SAMPLE_TEXTS[chosen_sample]
            st.text_area(
                "Sample text content (editable)",
                value=current_text,
                height=180,
                key="sample_text_area",
            )
            current_text = st.session_state.get("sample_text_area", current_text)

    elif input_mode == "Text input":
        current_text = st.text_area(
            "Enter or paste text to analyze",
            value="",
            height=200,
            placeholder="Type or paste an essay, article, or generated response here...",
            key="direct_text_input",
        )

    elif input_mode == "Upload PDF":
        pdf_file = st.file_uploader(
            "Upload a document (PDF)",
            type=["pdf"],
            help="Extracts text directly from digital PDF pages",
        )
        if pdf_file:
            with st.spinner("Parsing PDF document..."):
                extracted, meta = app_features.extract_text_from_pdf(pdf_file)
                input_metadata = meta
            if meta.get("is_scanned"):
                st.warning(
                    f"Notice: Extracted only {meta.get('extracted_chars', 0)} characters across {meta.get('pages', 0)} pages. "
                    "This PDF may be scanned or image-only.",
                    icon=":material/warning:",
                )
            else:
                st.success(
                    f"Extracted {meta.get('word_count', 0):,} words across {meta.get('pages', 0)} pages.",
                    icon=":material/check_circle:",
                )
            current_text = st.text_area(
                "Extracted PDF text (editable before classification)",
                value=extracted,
                height=200,
                key="pdf_extracted_area",
            )

    elif input_mode == "Upload image":
        img_file = st.file_uploader(
            "Upload an image containing text",
            type=["png", "jpg", "jpeg", "webp"],
            help="Applies OCR preprocessing and extracts text using Tesseract or EasyOCR",
        )
        if img_file:
            col_img, col_ocr = st.columns([1, 1], vertical_alignment="top")
            with col_img:
                st.image(
                    img_file,
                    caption=f"Uploaded: {img_file.name}",
                    alt=f"Uploaded image {img_file.name}",
                    width=400,
                )
            with col_ocr:
                with st.spinner("Executing Optical Character Recognition (OCR)..."):
                    try:
                        extracted, meta = app_features.extract_text_from_image(
                            img_file,
                            engine=ocr_engine,
                            tesseract_cmd=tess_path_input or None,
                        )
                        input_metadata = meta
                        st.success(
                            f"OCR completed via {meta.get('engine')} ({meta.get('word_count', 0)} words found).",
                            icon=":material/check_circle:",
                        )
                    except Exception as e:
                        st.error(f"OCR failed: {e}", icon=":material/error:")
                        extracted = ""
                current_text = st.text_area(
                    "Extracted image text (editable)",
                    value=extracted,
                    height=180,
                    key="image_extracted_area",
                )

    # Word & Character Count summary
    words_count = len(current_text.strip().split()) if current_text.strip() else 0
    chars_count = len(current_text.strip())

    col_btn, col_stats = st.columns([1, 2], vertical_alignment="center")
    with col_btn:
        run_btn = st.button(
            "Run 9-model classification",
            type="primary",
            disabled=(chars_count < 10 or not sel_feats or not sel_clfs),
            icon=":material/rocket_launch:",
            width="stretch",
        )
    with col_stats:
        if chars_count > 0:
            st.caption(f"Input size: **{words_count:,}** words | **{chars_count:,}** characters | Target models: **{len(sel_feats) * len(sel_clfs)}**")
        else:
            st.caption("Please input or upload at least 10 characters to run classification.")

    # ------------------------------------------------------------------------
    # EXECUTION & RESULTS
    # ------------------------------------------------------------------------
    if run_btn and current_text.strip():
        st.divider()
        st.subheader("2. Ensemble results & predictions", anchor=False)

        total_runs = len(sel_feats) * len(sel_clfs)
        results = []
        progress_bar = st.progress(0, text="Initializing model features...")

        step_counter = 0
        computed_features = {}

        # 1. Feature computation
        for feat in sel_feats:
            progress_bar.progress(
                int((step_counter / total_runs) * 100),
                text=f"Computing {FEATURE_SETS[feat]['title']} representation...",
            )
            try:
                computed_features[feat] = extract_feature_vector(
                    feat, current_text, extras, embed_name
                )
            except Exception as ex:
                st.error(f"Error extracting {FEATURE_SETS[feat]['title']}: {ex}")
                continue

            # 2. Classifier evaluation
            for clf in sel_clfs:
                step_counter += 1
                progress_bar.progress(
                    int((step_counter / total_runs) * 100),
                    text=f"Running {FEATURE_SETS[feat]['title']} + {CLASSIFIERS[clf]['title']}...",
                )
                model_key = (feat, clf)
                model_inst = models.get(model_key)
                if model_inst is None:
                    continue

                try:
                    label, conf, prob_dict = predict_single_model(
                        model_inst, computed_features[feat], label_map
                    )
                    results.append({
                        "feat_key": feat,
                        "clf_key": clf,
                        "Pipeline": FEATURE_SETS[feat]["title"],
                        "Classifier": CLASSIFIERS[clf]["title"],
                        "Model Combo": f"{FEATURE_SETS[feat]['title']} + {CLASSIFIERS[clf]['title']}",
                        "Prediction": label,
                        "Confidence": conf,
                        "P_Human": prob_dict.get(label_map.get("0", "Human-Written"), 0.0),
                        "P_AI": prob_dict.get(label_map.get("1", "AI-Generated"), 0.0),
                    })
                except Exception as ex:
                    st.error(f"Error predicting with {feat}__{clf}: {ex}")

        progress_bar.empty()

        if results:
            res_df = pd.DataFrame(results)

            # Majority Vote calculation
            vote_counts = res_df["Prediction"].value_counts()
            majority_label = vote_counts.idxmax()
            majority_count = vote_counts.max()
            consensus_pct = (majority_count / len(results)) * 100
            avg_conf = res_df["Confidence"].mean() * 100

            # Consensus Card
            with st.container(border=True):
                col_verdict, col_metrics1, col_metrics2 = st.columns([2, 1, 1], vertical_alignment="center")
                is_ai = majority_label == label_map.get("1", "AI-Generated")

                with col_verdict:
                    if is_ai:
                        st.markdown(f"### 🤖 Consensus: **{majority_label}**")
                        st.markdown(
                            f"**{majority_count} out of {len(results)}** models voted **{majority_label}** "
                            f"({consensus_pct:.1f}% agreement)."
                        )
                    else:
                        st.markdown(f"### ✍️ Consensus: **{majority_label}**")
                        st.markdown(
                            f"**{majority_count} out of {len(results)}** models voted **{majority_label}** "
                            f"({consensus_pct:.1f}% agreement)."
                        )

                with col_metrics1:
                    st.metric(
                        "Ensemble Agreement",
                        f"{consensus_pct:.1f}%",
                        f"{majority_count}/{len(results)} models",
                    )
                with col_metrics2:
                    st.metric(
                        "Mean Confidence",
                        f"{avg_conf:.1f}%",
                    )

            # 3x3 MODEL CARDS GRID
            st.markdown("#### Individual model predictions (3 $\\times$ 3 matrix)")

            for feat in sel_feats:
                st.markdown(f"**{FEATURE_SETS[feat]['icon']} {FEATURE_SETS[feat]['title']}** — *{FEATURE_SETS[feat]['desc']}*")
                cols = st.columns(len(sel_clfs))

                for idx, clf in enumerate(sel_clfs):
                    sub = res_df[(res_df["feat_key"] == feat) & (res_df["clf_key"] == clf)]
                    with cols[idx]:
                        with st.container(border=True):
                            st.caption(f"{CLASSIFIERS[clf]['icon']} **{CLASSIFIERS[clf]['title']}**")
                            if not sub.empty:
                                row = sub.iloc[0]
                                pred_str = row["Prediction"]
                                conf_val = row["Confidence"]
                                p_ai = row["P_AI"]
                                p_hum = row["P_Human"]

                                badge_color = "red" if pred_str == label_map.get("1", "AI-Generated") else "green"
                                st.markdown(f":{badge_color}-background[**{pred_str}**]")
                                st.caption(f"Confidence: **{conf_val:.1%}**")

                                # Visual probability split
                                st.progress(p_ai, text=f"AI: {p_ai:.1%} | Human: {p_hum:.1%}")
                            else:
                                st.write("Model skipped")

            # VISUAL COMPARISON CHART
            st.markdown("#### Probability breakdown across all models")
            chart_data = res_df[["Model Combo", "P_AI", "P_Human"]].copy()
            chart_data = chart_data.rename(columns={
                "P_AI": f"P({label_map.get('1', 'AI')})",
                "P_Human": f"P({label_map.get('0', 'Human')})",
            }).set_index("Model Combo")

            st.bar_chart(chart_data, height=320, width="stretch")

            # STYLOMETRIC & LINGUISTIC FEATURE INSPECTION
            with st.expander("Detailed linguistic feature breakdown", expanded=False, icon=":material/analytics:"):
                ling_dict = app_features.compute_linguistic_dict(current_text)
                ling_cols = st.columns(4)

                ling_metrics = [
                    ("Word Count", f"{int(ling_dict['word_count']):,}", "Total words"),
                    ("Avg Word Length", f"{ling_dict['avg_word_length']:.2f} chars", "Characters per word"),
                    ("Sentence Count", f"{int(ling_dict['sentence_count']):,}", "Total sentences"),
                    ("Avg Sentence Length", f"{ling_dict['avg_sentence_length']:.1f} words", "Words per sentence"),
                    ("Lexical Diversity (TTR)", f"{ling_dict['type_token_ratio']:.3f}", "Unique / total words"),
                    ("Punctuation Density", f"{ling_dict['punctuation_ratio']:.3f}", "Marks per character"),
                    ("Uppercase Density", f"{ling_dict['uppercase_ratio']:.3f}", "Uppercase per char"),
                    ("Digit Density", f"{ling_dict['digit_ratio']:.3f}", "Digits per char"),
                ]

                for i, (m_title, m_val, m_desc) in enumerate(ling_metrics):
                    with ling_cols[i % 4]:
                        st.metric(m_title, m_val, help=m_desc)

                scaler = extras.get("linguistic_scaler")
                if scaler is not None:
                    st.caption("Comparison to training set distribution (StandardScaler baseline):")
                    scaler_df = pd.DataFrame({
                        "Feature": app_features.ORDERED_FEATURES,
                        "Current Text": [ling_dict[f] for f in app_features.ORDERED_FEATURES],
                        "Training Mean": scaler.mean_,
                        "Training Std": scaler.scale_,
                    })
                    st.dataframe(scaler_df, width="stretch", hide_index=True)

            # EXPORT DATA
            with st.expander("Export prediction report", expanded=False, icon=":material/download:"):
                col_csv, col_json = st.columns(2)
                csv_bytes = res_df.to_csv(index=False).encode("utf-8")
                json_bytes = res_df.to_json(orient="records", indent=2).encode("utf-8")

                with col_csv:
                    st.download_button(
                        "Download predictions as CSV",
                        data=csv_bytes,
                        file_name="genai_predictions.csv",
                        mime="text/csv",
                        icon=":material/download:",
                        width="stretch",
                    )
                with col_json:
                    st.download_button(
                        "Download predictions as JSON",
                        data=json_bytes,
                        file_name="genai_predictions.json",
                        mime="application/json",
                        icon=":material/download:",
                        width="stretch",
                    )


# ============================================================================
# TAB 2: BENCHMARKS & PERFORMANCE (TEST DATASET RESULTS)
# ============================================================================
with tab_benchmarks:
    st.subheader("Model benchmark results on test datasets", anchor=False)
    st.markdown(
        "Evaluation metrics across **2,700 held-out test texts** and **1,350 edited AI texts** "
        "from multiple benchmark datasets (`artem9k_ai-text-detection-pile`, `inokusan_human_ai_text_classification`, `liamdugan_raid`)."
    )

    bench_csv = ART_DIR / "benchmarks.csv"
    bench_json = ART_DIR / "benchmarks.json"

    if bench_csv.exists():
        bdf = pd.read_csv(bench_csv)

        # Overview KPI metric cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        best_acc_row = bdf.loc[bdf["accuracy"].idxmax()]
        best_f1_row = bdf.loc[bdf["f1"].idxmax()]
        best_auc_row = bdf.loc[bdf["roc_auc"].idxmax()]
        best_edited_row = bdf.loc[bdf["edited_ai_recall"].idxmax()]

        with kpi1:
            st.metric(
                "Best Accuracy",
                f"{best_acc_row['accuracy']:.1%}",
                f"{best_acc_row['model_name']}",
            )
        with kpi2:
            st.metric(
                "Best F1 Score",
                f"{best_f1_row['f1']:.1%}",
                f"{best_f1_row['model_name']}",
            )
        with kpi3:
            st.metric(
                "Best ROC-AUC",
                f"{best_auc_row['roc_auc']:.3f}",
                f"{best_auc_row['model_name']}",
            )
        with kpi4:
            st.metric(
                "Best Edited AI Recall",
                f"{best_edited_row['edited_ai_recall']:.1%}",
                f"{best_edited_row['model_name']}",
            )

        st.divider()

        # Metric comparison chart
        st.markdown("#### Performance comparison across all 9 models")
        metric_choice = st.segmented_control(
            "Select evaluation metric",
            options=["accuracy", "f1", "roc_auc", "edited_ai_recall", "train_seconds"],
            default="accuracy",
            format_func=lambda x: {
                "accuracy": "Accuracy",
                "f1": "F1-Score",
                "roc_auc": "ROC-AUC",
                "edited_ai_recall": "Edited AI Recall",
                "train_seconds": "Training Time (s)",
            }[x],
        )

        chart_df = bdf[["model_name", metric_choice]].set_index("model_name")
        st.bar_chart(chart_df, height=340, width="stretch")

        st.markdown("#### Comprehensive benchmark summary table")
        display_cols = [
            "model_name", "feature_set", "classifier", "accuracy", "f1",
            "roc_auc", "precision", "recall", "fpr", "fnr", "edited_ai_recall", "train_seconds"
        ]
        available_cols = [c for c in display_cols if c in bdf.columns]
        formatted_bdf = bdf[available_cols].copy()
        st.dataframe(
            formatted_bdf.style.format({
                "accuracy": "{:.2%}",
                "f1": "{:.2%}",
                "roc_auc": "{:.3f}",
                "precision": "{:.2%}",
                "recall": "{:.2%}",
                "fpr": "{:.2%}",
                "fnr": "{:.2%}",
                "edited_ai_recall": "{:.2%}",
                "train_seconds": "{:.1f}s",
            }),
            width="stretch",
            hide_index=True,
        )

        # Per dataset breakdown
        with st.expander("Per-dataset breakdown", expanded=False, icon=":material/dataset:"):
            ds_cols = [c for c in bdf.columns if "__" in c]
            if ds_cols:
                ds_df = bdf[["model_name"] + ds_cols].set_index("model_name")
                st.dataframe(ds_df, width="stretch")

    elif bench_json.exists():
        with open(bench_json) as f:
            st.json(json.load(f))
    else:
        st.info("No benchmark file found in artifacts/ directory.")


# ============================================================================
# TAB 3: MODEL ARCHITECTURE & PIPELINE DETAILS
# ============================================================================
with tab_architecture:
    st.subheader("Architectural design: 3 Representations $\\times$ 3 Classifiers", anchor=False)

    col_repr, col_clf = st.columns(2)

    with col_repr:
        with st.container(border=True):
            st.markdown("### 🔤 Feature Representations (NLP)")

            st.markdown(
                "**1. TF-IDF N-grams (Sparse Lexical)**\n"
                "- **Dimensionality:** 20,000 features\n"
                "- **Mechanism:** Word and subword n-gram frequencies weighted by inverse document frequency.\n"
                "- **Key advantage:** Captures distinctive phrases, formal transition words, and vocabulary specific to LLMs."
            )
            st.divider()
            st.markdown(
                "**2. Dense Sentence Embeddings (Contextual)**\n"
                "- **Dimensionality:** 384 dimensions (`all-MiniLM-L6-v2`)\n"
                "- **Mechanism:** Transformer encoder pooling continuous latent semantic spaces.\n"
                "- **Key advantage:** Evaluates topic coherence, semantic continuity, and sentence embeddings."
            )
            st.divider()
            st.markdown(
                "**3. Stylometric & Linguistic (Syntax)**\n"
                "- **Dimensionality:** 10 hand-engineered features + standard scaling\n"
                "- **Features:** Character count, word count, avg word length, sentence count, comma frequency, "
                "punctuation count & ratio, uppercase ratio, digit ratio, and Type-Token Ratio (lexical diversity).\n"
                "- **Key advantage:** Highly resistant to topic distribution shifts and captures the uniform pacing of LLMs."
            )

    with col_clf:
        with st.container(border=True):
            st.markdown("### ⚙️ Machine Learning Classifiers")

            st.markdown(
                "**1. Logistic Regression**\n"
                "- **Type:** Convex linear decision boundary with L2 regularization.\n"
                "- **Inference:** Fast, highly interpretable sigmoid probabilities."
            )
            st.divider()
            st.markdown(
                "**2. Random Forest**\n"
                "- **Type:** Non-linear ensemble of bagged decision trees.\n"
                "- **Inference:** Exceptional performance on stylometric features (81.6% accuracy with only 10 features)."
            )
            st.divider()
            st.markdown(
                "**3. Support Vector Machine (Linear SVM)**\n"
                "- **Type:** Maximum-margin linear hyperplane separator.\n"
                "- **Calibration:** Calibrated using Platt scaling / sigmoid calibration (`CalibratedClassifierCV`) "
                "for well-calibrated confidence scores. Top performer on TF-IDF (85.7% accuracy, 93.3% ROC-AUC)."
            )

    with st.container(border=True):
        st.markdown("### 🤝 Ensemble Strategy")
        st.markdown(
            "By taking a **majority vote** across the 9 combinations, the system balances lexical n-gram signals "
            "(TF-IDF), semantic contextual embeddings (SentenceTransformers), and syntactic cadence (Linguistic). "
            "When all 9 models agree, detection confidence is exceptionally high; when models diverge, "
            "the confidence distribution highlights borderline or human-edited text."
        )
