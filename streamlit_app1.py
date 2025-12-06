# import streamlit as st
# import google.generativeai as genai
# from dotenv import load_dotenv
# import os
# import io
# from PIL import Image
# import traceback

# # ---------------------------------------------------------
# # SAFE APP WRAPPER (prevents blank screen)
# # ---------------------------------------------------------
# def safe_run(func):
#     try:
#         func()
#     except Exception:
#         st.error("🚨 An error occurred! See details below:")
#         st.code(traceback.format_exc())


# # ---------------------------------------------------------
# # LOAD .env (DO NOT STOP IF NOT FOUND)
# # ---------------------------------------------------------
# load_dotenv()

# API_KEY = os.getenv("GEMINI_API_KEY")
# if not API_KEY:
#     st.error("Some error occured")
# else:
#     genai.configure(api_key=API_KEY)


# # ---------------------------------------------------------
# # GEMINI MODEL (Vision-enabled)
# # ---------------------------------------------------------
# MODEL_NAME = "gemini-2.5-flash"   # stable + supports images

# model = genai.GenerativeModel(MODEL_NAME)


# # ---------------------------------------------------------
# # OCR USING GEMINI (multimodal)
# # ---------------------------------------------------------
# def gemini_extract_text(uploaded_file):
#     try:
#         img = uploaded_file.read()
#         image = Image.open(io.BytesIO(img))

#         prompt = """
#         Extract ALL text from this medical document clearly.
#         Maintain line breaks. Do NOT summarize.
#         """

#         response = model.generate_content([prompt, image])
#         return response.text.strip()

#     except Exception as e:
#         return f"OCR error: {str(e)}"


# # ---------------------------------------------------------
# # ANALYZE REPORT
# # ---------------------------------------------------------
# def analyze_report(text):
#     prompt = f"""
#     You are a medical REPORT analyzer.

#     From the extracted text, provide:
#     1. A clean table of test names, values, and normal ranges (if present).
#     2. Mark each value as HIGH / LOW / NORMAL.
#     3. Explain abnormalities in simple language.
#     4. Give a final doctor-style summary.

#     REPORT TEXT:
#     {text}
#     """

#     try:
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"Report analysis failed: {str(e)}"


# # ---------------------------------------------------------
# # ANALYZE PRESCRIPTION
# # ---------------------------------------------------------
# def analyze_prescription(text):
#     prompt = f"""
#     You are a medical PRESCRIPTION analyzer.

#     Extract and return:
#     1. Medicine Name
#     2. Strength (mg)
#     3. Dose and Frequency (OD, BD, TID, QHS, etc.)
#     4. Timing (Before Food / After Food)
#     5. Duration (if any)
#     6. Purpose of each medicine in simple language

#     PRESCRIPTION TEXT:
#     {text}
#     """

#     try:
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"Prescription analysis failed: {str(e)}"


# # ---------------------------------------------------------
# # AUTO-DETECT TYPE & ANALYZE
# # ---------------------------------------------------------
# def auto_analyze(text):
#     t = text.lower()

#     pres_keywords = ["tab", "tablet", "mg", "ml", "od", "bd", "tid", "dose"]
#     report_keywords = ["hemoglobin", "wbc", "range", "value", "bilirubin", "test"]

#     pres_score = sum(k in t for k in pres_keywords)
#     rep_score = sum(k in t for k in report_keywords)

#     if pres_score > rep_score:
#         return "Prescription", analyze_prescription(text)
#     else:
#         return "Medical Report", analyze_report(text)


# # ---------------------------------------------------------
# # STREAMLIT UI
# # ---------------------------------------------------------
# def main():
#     st.set_page_config(page_title="Medical Analyzer", layout="wide")
#     st.title("🩺 Medical Report & Prescription Analyzer")

#     uploaded = st.file_uploader("Upload Image/PDF", type=["png", "jpg", "jpeg", "pdf"])

#     if uploaded:
#         st.subheader("🔍 Extracting text with Donut…")
#         extracted_text = gemini_extract_text(uploaded)

#         st.text_area("Extracted Text", extracted_text, height=250)

#         if extracted_text.startswith("OCR error"):
#             st.error(extracted_text)
#             return

#         st.subheader("📘 Detecting Document Type…")
#         doc_type, analysis = auto_analyze(extracted_text)

#         st.success(f"Detected Document Type: **{doc_type}**")

#         st.subheader("📊 Analysis")
#         st.write(analysis)

#         # Download JSON
#         st.download_button(
#             "Download Analysis as Text",
#             analysis,
#             file_name=f"{doc_type.replace(' ', '_').lower()}_analysis.txt"
#         )

#     else:
#         st.info("Upload a medical report or prescription to begin.")

# safe_run(main)





# # streamlit_app.py
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import io
from PIL import Image
import traceback
from gtts import gTTS
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import tempfile

# ---------------------------------------------------------
# SAFE APP WRAPPER (prevents blank screen)
# ---------------------------------------------------------
def safe_run(func):
    try:
        func()
    except Exception:
        st.error("🚨 An error occurred! See details below:")
        st.code(traceback.format_exc())


# ---------------------------------------------------------
# LOAD .env (DO NOT STOP IF NOT FOUND)
# ---------------------------------------------------------
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("Some error occurred")
else:
    genai.configure(api_key=API_KEY)


# ---------------------------------------------------------
# GEMINI MODEL (Vision-enabled)
# ---------------------------------------------------------
MODEL_NAME = "gemini-2.5-flash"

model = genai.GenerativeModel(MODEL_NAME)


# ---------------------------------------------------------
# OCR USING GEMINI (multimodal)
# ---------------------------------------------------------
def gemini_extract_text(uploaded_file):
    try:
        img = uploaded_file.read()
        image = Image.open(io.BytesIO(img))

        prompt = """
        Extract ALL text from this medical document clearly.
        Maintain line breaks. Do NOT summarize.
        """

        response = model.generate_content([prompt, image])
        return response.text.strip()

    except Exception as e:
        return f"OCR error: {str(e)}"


# ---------------------------------------------------------
# ANALYZE REPORT
# ---------------------------------------------------------
def analyze_report(text, language="English"):
    lang_instruction = ""
    if language == "Kannada":
        lang_instruction = "Provide the ENTIRE response in Kannada language (ಕನ್ನಡ)."
    
    prompt = f"""
    You are a medical REPORT analyzer.

    From the extracted text, provide:
    1. A clean table of test names, values, normal ranges, and status (HIGH/LOW/NORMAL).
    2. After the table, list ONLY the abnormal values (HIGH or LOW) with a single line explanation on how to improve it.

    Format:
    [TABLE HERE]
    
    **Abnormal Values:**
    - Test Name (Status): Brief one-line advice to improve it.

    Keep it concise and actionable. Do NOT include any summary section.
    {lang_instruction}

    REPORT TEXT:
    {text}
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Report analysis failed: {str(e)}"


# ---------------------------------------------------------
# ANALYZE PRESCRIPTION
# ---------------------------------------------------------
def analyze_prescription(text, language="English"):
    lang_instruction = ""
    if language == "Kannada":
        lang_instruction = "Provide the ENTIRE response in Kannada language (ಕನ್ನಡ)."
    
    prompt = f"""
    You are a medical PRESCRIPTION analyzer.

    Extract and return:
    1. Medicine Name
    2. Strength (mg)
    3. Dose and Frequency (OD, BD, TID, QHS, etc.)
    4. Timing (Before Food / After Food)
    5. Duration (if any)
    6. Purpose of each medicine in simple language

    {lang_instruction}

    PRESCRIPTION TEXT:
    {text}
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Prescription analysis failed: {str(e)}"


# ---------------------------------------------------------
# AUTO-DETECT TYPE & ANALYZE
# ---------------------------------------------------------
def auto_analyze(text, language="English"):
    t = text.lower()

    pres_keywords = ["tab", "tablet", "mg", "ml", "od", "bd", "tid", "dose"]
    report_keywords = ["hemoglobin", "wbc", "range", "value", "bilirubin", "test"]

    pres_score = sum(k in t for k in pres_keywords)
    rep_score = sum(k in t for k in report_keywords)

    if pres_score > rep_score:
        return "Prescription", analyze_prescription(text, language)
    else:
        return "Medical Report", analyze_report(text, language)


# ---------------------------------------------------------
# TEXT TO SPEECH
# ---------------------------------------------------------
def text_to_speech(text, language="English"):
    try:
        lang_code = "kn" if language == "Kannada" else "en"
        tts = gTTS(text=text, lang=lang_code, slow=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"Voice generation error: {str(e)}")
        return None


# ---------------------------------------------------------
# GENERATE PDF
# ---------------------------------------------------------
def generate_pdf(content, doc_type):
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        
        styles = getSampleStyleSheet()
        
        # Title style
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor='darkblue',
            spaceAfter=30,
            alignment=TA_LEFT
        )
        
        # Body style
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            leading=16,
            alignment=TA_LEFT
        )
        
        # Add title
        title = Paragraph(f"Medical Analysis: {doc_type}", title_style)
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Add content (split by lines and create paragraphs)
        for line in content.split('\n'):
            if line.strip():
                # Replace special characters that might cause issues
                safe_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                para = Paragraph(safe_line, body_style)
                story.append(para)
                story.append(Spacer(1, 6))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"PDF generation error: {str(e)}")
        return None


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Medical Analyzer", layout="wide")
    st.title("🩺 Medical Report & Prescription Analyzer")
    
    # Language selection
    col1, col2 = st.columns([3, 1])
    with col2:
        language = st.selectbox(
            "Output Language",
            ["English", "Kannada"],
            help="Select the language for analysis output"
        )

    uploaded = st.file_uploader("Upload Image/PDF", type=["png", "jpg", "jpeg", "pdf"])

    if uploaded:
        with st.spinner("🔍 Extracting text..."):
            extracted_text = gemini_extract_text(uploaded)

        if extracted_text.startswith("OCR error"):
            st.error(extracted_text)
            return

        with st.spinner("📘 Analyzing document..."):
            doc_type, analysis = auto_analyze(extracted_text, language)

        st.success(f"✅ Detected Document Type: **{doc_type}**")

        st.subheader("📊 Analysis")
        st.write(analysis)

        # Voice output
        st.subheader("🔊 Audio Output")
        audio_file = text_to_speech(analysis, language)
        if audio_file:
            with open(audio_file, "rb") as audio:
                st.audio(audio.read(), format="audio/mp3")
            os.unlink(audio_file)  # Clean up temp file

        # Download buttons
        st.subheader("📥 Download Options")
        col1, col2 = st.columns(2)
        
        with col1:
            # Text download
            st.download_button(
                "📄 Download as Text",
                analysis,
                file_name=f"{doc_type.replace(' ', '_').lower()}_analysis.txt",
                mime="text/plain"
            )
        
        with col2:
            # PDF download
            pdf_buffer = generate_pdf(analysis, doc_type)
            if pdf_buffer:
                st.download_button(
                    "📑 Download as PDF",
                    pdf_buffer,
                    file_name=f"{doc_type.replace(' ', '_').lower()}_analysis.pdf",
                    mime="application/pdf"
                )

    else:
        st.info("📤 Upload a medical report or prescription to begin.")

safe_run(main)