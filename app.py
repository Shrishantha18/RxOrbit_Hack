# gemini_ocr_and_analyze.py
import os
import io
import json
import traceback
from PIL import Image
from pathlib import Path

# Try both possible GenAI client import names (some users have different packages)
try:
    from google import genai as genai_client
    GENAI_PKG = "google.genai"
except Exception:
    try:
        import google.generativeai as genai_client
        GENAI_PKG = "google.generativeai"
    except Exception:
        genai_client = None
        GENAI_PKG = None

# Optional local fallback OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    TESSERACT_AVAILABLE = False

# PDF->image helper
try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

# Load API key from env (streamlit can set env or st.secrets will be used by caller)
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API") or None

def init_client(api_key=None):
    """Initialize gemini client. Returns client or None."""
    key = api_key or API_KEY
    if genai_client is None:
        return None
    try:
        # different libs have different client shapes
        if hasattr(genai_client, "Client"):
            return genai_client.Client(api_key=key)
        else:
            # older google.generativeai style
            genai_client.configure(api_key=key)
            return genai_client
    except Exception:
        return None

# Single helper to convert uploaded streamlit file to PIL.Image (first page for PDFs)
def uploaded_file_to_pil(uploaded_file):
    uploaded_file.seek(0)
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    if name.endswith(".pdf"):
        if convert_from_bytes is None:
            raise RuntimeError("pdf2image/poppler not available for PDF conversion.")
        pages = convert_from_bytes(raw, dpi=300, fmt="png")
        img = pages[0].convert("RGB")
        return img
    else:
        return Image.open(io.BytesIO(raw)).convert("RGB")

# Gemini multimodal image->text extraction wrapper
def gemini_extract_text_from_image(client, pil_image, model="gemini-2.5-flash", instruction=None, timeout_seconds=60):
    """
    Send PIL image + instruction to Gemini and return extracted text (string) + raw response.
    """
    if client is None:
        raise RuntimeError("Gemini client not initialized")

    if instruction is None:
        instruction = (
            "Extract all textual content from the image exactly as it appears. "
            "Return only the text (preserve line breaks). If possible, also return classification under "
            "'[TYPE:] prescription' or '[TYPE:] report' at the top. Do not add extra commentary."
        )

    # Some client libs accept pillow image directly in contents; others accept bytes.
    contents = [instruction, pil_image]
    try:
        # new-style client: client.models.generate_content(...)
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                temperature=0.0,
                max_output_tokens=1024,
                timeout=timeout_seconds
            )
            # Extract textual parts
            text_parts = []
            try:
                cand = resp.candidates[0]
                for p in cand.content.parts:
                    if getattr(p, "text", None):
                        text_parts.append(p.text)
                    elif getattr(p, "inline_text", None):  # some SDK variants
                        text_parts.append(p.inline_text)
            except Exception:
                # fallback to str(resp)
                text_parts = [str(resp)]
            full_text = "\n".join([t.strip() for t in text_parts if t and t.strip()])
            return full_text, resp
        else:
            # older direct client
            out = client.generate(prompt=instruction, image=pil_image)
            return out, None
    except Exception as e:
        raise RuntimeError(f"Gemini image->text failed: {e}")

# Fallback local OCR
def local_tesseract_extract(pil_image):
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        gray = pil_image.convert("L")
        txt = pytesseract.image_to_string(gray, lang="eng", config="--psm 6")
        return txt or ""
    except Exception:
        return ""

# Analysis wrappers: for short, human-readable outputs (B)
def analyze_prescription_text(client, text, model="gemini-2.5-flash", language="english"):
    """
    Ask Gemini to produce a short human-readable prescription summary.
    language: "english" or "kannada" (the model will be asked to produce requested language)
    Output: short bullet list per medicine; if strength/frequency missing, omit those fields.
    """
    if client is None:
        raise RuntimeError("Gemini client not initialized")

    lang_tag = "English" if language.lower().startswith("en") else "Kannada"
    prompt = f"""
You are a prescription parser. Produce a **short** human-readable list of prescribed medicines.
For each medicine include:
- Name
- Strength (if present)
- Frequency/dose (if explicitly present; otherwise omit)
- One-line typical purpose (one short phrase)
Return the result as short bullets. Use {lang_tag}. Keep the output concise (no long paragraphs).
Prescription text:
{text}
"""
    try:
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            resp = client.models.generate_content(
                model=model,
                contents=[prompt],
                temperature=0.0,
                max_output_tokens=500
            )
            out = ""
            cand = resp.candidates[0]
            for p in cand.content.parts:
                if getattr(p, "text", None):
                    out += p.text
            return out.strip(), resp
        else:
            out = client.generate(prompt=prompt)
            return out, None
    except Exception as e:
        return f"Prescription analysis failed: {e}", None

def analyze_report_text(client, text, model="gemini-2.5-flash", language="english"):
    """
    Short human-readable report output:
    - For each lab/test: Test Name — value — status (HIGH/LOW/NORMAL) — one-line note (concise)
    - End with 1-2 line patient-friendly summary in requested language.
    """
    if client is None:
        raise RuntimeError("Gemini client not initialized")

    lang_tag = "English" if language.lower().startswith("en") else "Kannada"
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
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            resp = client.models.generate_content(
                model=model,
                contents=[prompt],
                temperature=0.0,
                max_output_tokens=700
            )
            out = ""
            cand = resp.candidates[0]
            for p in cand.content.parts:
                if getattr(p, "text", None):
                    out += p.text
            return out.strip(), resp
        else:
            out = client.generate(prompt=prompt)
            return out, None
    except Exception as e:
        return f"Report analysis failed: {e}", None

# High-level orchestrator used by Streamlit app
def extract_and_analyze(uploaded_file, gemini_model_name="gemini-2.5-flash", prefer_gemini_for_ocr=True, language="english"):
    """
    Returns:
    {
      'extracted_text': str,
      'extractor': 'gemini'|'tesseract'|'none',
      'doc_type': 'prescription'|'report'|'unknown',
      'analysis_text': str,           # short, in requested language
      'analysis_text_english_for_pdf': str,  # always English for PDF output
      'raw_extraction_resp': obj or None,
      'raw_analysis_resp': obj or None
    }
    """
    client = init_client()
    pil_img = None
    try:
        pil_img = uploaded_file_to_pil(uploaded_file)
    except Exception as e:
        # try to proceed with tesseract if possible
        pil_img = None

    extracted_text = ""
    extractor_used = "none"
    raw_extract_resp = None

    # Try Gemini OCR first
    if prefer_gemini_for_ocr and client is not None and pil_img is not None:
        try:
            instruction = (
                "Extract all text from the image and if possible, at the top include a line with '[TYPE:] prescription' or '[TYPE:] report' "
                "if you can detect the document type. Return only the extracted text (no commentary)."
            )
            extracted_text, raw_extract_resp = gemini_extract_text_from_image(client, pil_img, model=gemini_model_name, instruction=instruction)
            if extracted_text and extracted_text.strip():
                extractor_used = "gemini"
        except Exception:
            extracted_text = ""
            extractor_used = "none"

    # Fallback to local Tesseract
    if (not extracted_text or not extracted_text.strip()) and pil_img is not None:
        try:
            ttxt = local_tesseract_extract(pil_img)
            if ttxt and ttxt.strip():
                extracted_text = ttxt
                extractor_used = "tesseract"
        except Exception:
            pass

    # Final fallback nothing
    if not extracted_text:
        extracted_text = ""
        extractor_used = extractor_used or "none"

    # Detect document type (light heuristic). If Gemini included [TYPE:] line attempt to parse it
    doc_type = "unknown"
    if extracted_text:
        # check for explicit tag
        first_line = extracted_text.strip().splitlines()[0].lower() if extracted_text.strip() else ""
        if first_line.startswith("[type:]"):
            if "prescription" in first_line:
                doc_type = "prescription"
            elif "report" in first_line:
                doc_type = "report"
        else:
            t = extracted_text.lower()
            pres_k = ["tab", "tablet", "mg", "bd", "od", "tid", "dose", "take"]
            rep_k = ["hemoglobin", "wbc", "platelet", "rbc", "bilirubin", "creatinine", "glucose", "range", "reference", "result"]
            pres_score = sum(k in t for k in pres_k)
            rep_score = sum(k in t for k in rep_k)
            if pres_score >= rep_score + 1:
                doc_type = "prescription"
            elif rep_score >= pres_score + 1:
                doc_type = "report"
            else:
                doc_type = "unknown"

    # Run analysis according to detected doc_type
    analysis_text = ""
    analysis_resp = None
    analysis_text_english = ""

    try:
        client = init_client()  # re-init if needed
        if doc_type == "prescription":
            analysis_text, analysis_resp = analyze_prescription_text(client, extracted_text, model=gemini_model_name, language=language)
            # also produce english version for PDF
            if language.lower().startswith("en"):
                analysis_text_english = analysis_text
            else:
                analysis_text_english, _ = analyze_prescription_text(client, extracted_text, model=gemini_model_name, language="english")
        elif doc_type == "report":
            analysis_text, analysis_resp = analyze_report_text(client, extracted_text, model=gemini_model_name, language=language)
            # english for PDF
            if language.lower().startswith("en"):
                analysis_text_english = analysis_text
            else:
                analysis_text_english, _ = analyze_report_text(client, extracted_text, model=gemini_model_name, language="english")
        else:
            # ask Gemini to decide and provide short analysis in requested language
            combined_prompt = f"""
Decide whether the text is a prescription or report. Then provide a concise, short analysis (bulleted) in the requested language: { 'English' if language.lower().startswith('en') else 'Kannada' }.
Text:
{extracted_text}
"""
            if client is not None:
                resp = client.models.generate_content(model=gemini_model_name, contents=[combined_prompt], temperature=0.0, max_output_tokens=700)
                out = ""
                cand = resp.candidates[0]
                for p in cand.content.parts:
                    if getattr(p, "text", None):
                        out += p.text
                analysis_text = out.strip()
                # english for PDF
                resp2 = client.models.generate_content(model=gemini_model_name, contents=[combined_prompt.replace("Kannada", "English")], temperature=0.0, max_output_tokens=700)
                out2 = ""
                cand2 = resp2.candidates[0]
                for p in cand2.content.parts:
                    if getattr(p, "text", None):
                        out2 += p.text
                analysis_text_english = out2.strip()
            else:
                analysis_text = "No analysis available (client not configured)"
                analysis_text_english = analysis_text

    except Exception as e:
        analysis_text = f"Analysis failed: {e}"
        analysis_text_english = analysis_text

    return {
        "extracted_text": extracted_text,
        "extractor": extractor_used,
        "doc_type": doc_type,
        "analysis_text": analysis_text,
        "analysis_text_english_for_pdf": analysis_text_english,
        "raw_extraction_resp": raw_extract_resp,
        "raw_analysis_resp": analysis_resp
    }
