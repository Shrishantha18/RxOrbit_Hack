import streamlit as st
from app import extract_and_analyze

st.set_page_config(page_title="RxOrbit GRPO Test", layout="centered")

st.title("🧠 RxOrbit – OCR + GRPO Analyzer Test")

st.markdown("Upload a **prescription image** or **lab report PDF** to test the pipeline.")

# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.header("⚙️ Settings")

language = st.sidebar.selectbox(
    "Select Output Language",
    ["english", "kannada"]
)

use_grpo_for_prescription = st.sidebar.checkbox(
    "Use GRPO for Prescription",
    value=True
)

use_grpo_for_report = st.sidebar.checkbox(
    "Use GRPO for Report",
    value=True
)

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader(
    "Upload Prescription / Report (Image or PDF)",
    type=["png", "jpg", "jpeg", "pdf"]
)

# ------------------------------
# Run Analysis
# ------------------------------
if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    if st.button("🔍 Analyze Document"):
        with st.spinner("Processing OCR + GRPO Analysis..."):
            try:
                result = extract_and_analyze(
                    uploaded_file,
                    language=language,
                    use_grpo_for_prescription=use_grpo_for_prescription,
                    use_grpo_for_report=use_grpo_for_report,
                )

                # ------------------------------
                # Display Results
                # ------------------------------
                st.subheader("📄 Detected Document Type")
                st.info(result.get("doc_type", "unknown"))

                st.subheader("📝 Extracted Text")
                st.text_area(
                    "Raw OCR Output",
                    result.get("extracted_text", ""),
                    height=200
                )

                st.subheader("🧠 AI Analysis Output")
                st.markdown(result.get("analysis_text", "No analysis output"))

                st.subheader("📑 English Output for PDF")
                st.markdown(result.get("analysis_text_english_for_pdf", ""))

                with st.expander("⚙️ Debug / Raw API Responses"):
                    st.json({
                        "extractor": result.get("extractor"),
                        "raw_extraction_resp": str(result.get("raw_extraction_resp")),
                        "raw_analysis_resp": result.get("raw_analysis_resp"),
                    })

            except Exception as e:
                st.error("❌ Something went wrong during analysis")
                st.exception(e)

else:
    st.warning("⬆️ Please upload a file to start analysis.")
