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

# =========================
# Universal Gemini Text Generator (NEW + OLD SDK COMPATIBLE)
# =========================
def gemini_generate(client, prompt, model="gemini-2.5-flash", temperature=0.0, max_output_tokens=700):
    """
    Works with BOTH:
    - google.genai (new SDK)
    - google.generativeai (old SDK)
    """
    # ✅ New SDK style
    if hasattr(client, "models") and hasattr(client.models, "generate_content"):
        resp = client.models.generate_content(
            model=model,
            contents=[prompt],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        out = ""
        cand = resp.candidates[0]
        for p in cand.content.parts:
            if getattr(p, "text", None):
                out += p.text
        return out.strip(), resp

    # ✅ Old SDK style
    else:
        model_obj = client.GenerativeModel(model)
        resp = model_obj.generate_content(prompt)
        return resp.text.strip(), resp


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
# Gemini multimodal image->text extraction wrapper
def gemini_extract_text_from_image(
    client,
    pil_image,
    model="gemini-2.5-flash",
    instruction=None,
    timeout_seconds=60,
):
    """
    Send PIL image + instruction to Gemini and return extracted text (string) + raw response.
    Works with both google.genai and google.generativeai.
    """
    if client is None:
        raise RuntimeError("Gemini client not initialized")

    if instruction is None:
        instruction = (
            "Extract all textual content from the image exactly as it appears. "
            "Return only the text (preserve line breaks). If possible, also return classification under "
            "'[TYPE:] prescription' or '[TYPE:] report' at the top. Do not add extra commentary."
        )

    try:
        # ✅ New SDK: client.models.generate_content
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            resp = client.models.generate_content(
                model=model,
                contents=[instruction, pil_image],
                temperature=0.0,
                max_output_tokens=1024,
                timeout=timeout_seconds,
            )
            text_parts = []
            cand = resp.candidates[0]
            for p in cand.content.parts:
                if getattr(p, "text", None):
                    text_parts.append(p.text)
            full_text = "\n".join(text_parts).strip()
            return full_text, resp

        # ✅ Old SDK: google.generativeai
        else:
            model_obj = client.GenerativeModel(model)
            resp = model_obj.generate_content([instruction, pil_image])

            # old SDK usually has resp.text
            if hasattr(resp, "text"):
                return resp.text.strip(), resp
            else:
                return str(resp), resp

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
def analyze_prescription_text(client, text, model="gemini-2.5-flash", language="english", temperature=0.0):
    """
    Ask Gemini to produce a short human-readable prescription summary.
    Output will be strictly in English or Kannada as requested.
    """
    if client is None:
        raise RuntimeError("Gemini client not initialized")

    # HARD language lock
    if language.lower().startswith("en"):
        lang_rule = "You MUST respond ONLY in English. Do NOT use Kannada."
    else:
        lang_rule = "You MUST respond ONLY in Kannada. Do NOT use English."

    prompt = f"""
You are a strict medical prescription parser.

{lang_rule}

Produce a **short** human-readable list of prescribed medicines.

For each medicine include ONLY these fields (omit any field if not explicitly mentioned):
- Name
- Strength (if present)
- Frequency/dose (if explicitly present)
- Timing with respect to food (ONLY if explicitly mentioned, e.g., "Before Food" or "After Food")
- One-line typical purpose (one short phrase)

Return the result as short bullets in this exact format:

- Medicine Name | Strength | Frequency | Before Food / After Food | Purpose

Rules:
- If food timing is NOT mentioned in the prescription, OMIT that field completely.
- Do NOT guess food timing.
- Do NOT add extra explanations.
- Do NOT introduce new medicines.
- Keep output concise. No paragraphs.

Prescription text:
{text}
"""

    try:
        # main generation
        final_output, resp = gemini_generate(
            client,
            prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=500,
        )

        # ✅ SAFETY LANGUAGE CHECK (fallback)
        if language.lower().startswith("en"):
            # If Kannada accidentally appears, auto-retranslate to English
            if any("\u0C80" <= ch <= "\u0CFF" for ch in final_output):
                retry_prompt = f"Translate this strictly to English:\n{final_output}"
                final_output, _ = gemini_generate(
                    client,
                    retry_prompt,
                    model=model,
                    temperature=0.0,
                    max_output_tokens=300,
                )
        else:
            # If English appears, auto-retranslate to Kannada
            if any("a" <= ch.lower() <= "z" for ch in final_output):
                retry_prompt = f"Translate this strictly to Kannada:\n{final_output}"
                final_output, _ = gemini_generate(
                    client,
                    retry_prompt,
                    model=model,
                    temperature=0.0,
                    max_output_tokens=300,
                )

        return final_output.strip(), resp

    except Exception as e:
        return f"Prescription analysis failed: {e}", None


# =========================
# GRPO-style inference layer
# =========================

def grpo_generate_prescription_candidates(
    client,
    text,
    model="gemini-2.5-flash",
    language="english",
    num_candidates=2,
):
    """
    Policy step: generate multiple candidate analyses for the same prescription text.
    Uses higher temperature for diversity.
    """
    candidates = []
    for _ in range(num_candidates):
        out, resp = analyze_prescription_text(
            client,
            text,
            model=model,
            language=language,
            temperature=0.7,  # more diverse
        )
        candidates.append({"text": out, "raw_resp": resp})
    return candidates


def grpo_score_prescription_candidate(
    client,
    original_text,
    candidate_text,
    model="gemini-2.5-flash",
):
    """
    Reward model: ask Gemini to grade the quality of a candidate output.
    Returns a float score (0-10).
    """
    if client is None:
        return 0.0

    judge_prompt = f"""
You are grading the quality of a prescription summary.

Prescription text:
{original_text}

Candidate summary:
{candidate_text}

Give a score from 0 to 10 based on:
- Correct extraction of medicines and details
- Clear bullet formatting
- No hallucinated medicines
- Concise and easy to read

Return ONLY a number between 0 and 10. Do not add any words.
"""
    try:
        raw, _ = gemini_generate(
            client,
            judge_prompt,
            model=model,
            temperature=0.0,
            max_output_tokens=10,
        )
        score_str = raw.strip().split()[0]
        score = float(score_str)
        if score < 0:
            score = 0.0
        if score > 10:
            score = 10.0
        return score
    except Exception:
        return 0.0



def grpo_optimize_prescription(
    client,
    text,
    model="gemini-2.5-flash",
    language="english",
    num_candidates=2,
):
    """
    GRPO-style selection:
    1. Generate multiple candidates.
    2. Score each with a reward model.
    3. Return the highest-scoring candidate.
    """
    candidates = grpo_generate_prescription_candidates(
        client,
        text,
        model=model,
        language=language,
        num_candidates=num_candidates,
    )

    scored = []
    for c in candidates:
        s = grpo_score_prescription_candidate(
            client,
            original_text=text,
            candidate_text=c["text"],
            model=model,
        )
        scored.append({"score": s, "text": c["text"], "raw_resp": c["raw_resp"]})

    if not scored:
        return "No analysis available (GRPO scoring failed)", None

    # pick best scored candidate
    best = max(scored, key=lambda x: x["score"])
    # You could also return all scored candidates in raw_analysis_resp if you want
    return best["text"], {"scored_candidates": scored}



def analyze_report_text(client, text, model="gemini-2.5-flash", language="english", temperature=0.0):
    """
    Short human-readable report output:
    - For each lab/test: Test Name — value — normal range — status (HIGH/LOW/NORMAL)
    - After the table, list ONLY abnormal values with one-line advice.
    """
    if client is None:
        raise RuntimeError("Gemini client not initialized")

    if language.lower().startswith("en"):
        lang_instruction = "Write the entire output ONLY in English."
    else:
        lang_instruction = "Write the entire output ONLY in Kannada."

    prompt = f"""
You are a medical REPORT analyzer.

From the extracted lab report text, provide:

1. A clean table of test names, values, normal ranges, and status (HIGH/LOW/NORMAL).
2. After the table, list ONLY the abnormal values (HIGH or LOW) with a single line explanation on how to improve it.

Format:
[TABLE HERE]

**Abnormal Values:**
- Test Name (Status): Brief one-line advice to improve it.

Rules:
- Be concise and actionable.
- Do NOT include any separate summary section.
- {lang_instruction}

REPORT TEXT:
{text}
"""

    try:
        out, resp = gemini_generate(
            client,
            prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=700,
        )
        return out.strip(), resp

    except Exception as e:
        return f"Report analysis failed: {e}", None


# =========================
# GRPO-style inference layer for REPORTS
# =========================

def grpo_generate_report_candidates(
    client,
    text,
    model="gemini-2.5-flash",
    language="english",
    num_candidates=2,
):
    """
    Policy step: generate multiple candidate analyses for the same report text.
    Uses higher temperature for diversity.
    """
    candidates = []
    for _ in range(num_candidates):
        out, resp = analyze_report_text(
            client,
            text,
            model=model,
            language=language,
            temperature=0.7,  # more diverse
        )
        candidates.append({"text": out, "raw_resp": resp})
    return candidates


def grpo_score_report_candidate(
    client,
    original_text,
    candidate_text,
    model="gemini-2.5-flash",
):
    """
    Reward model: ask Gemini to grade the quality of a report summary.
    Returns a float score (0-10).
    """
    if client is None:
        return 0.0

    judge_prompt = f"""
You are grading the quality of a lab report analysis.

Lab report text:
{original_text}

Candidate analysis:
{candidate_text}

Give a score from 0 to 10 based on:
- Correct identification of test names and values
- Reasonable normal range/use of HIGH/LOW/NORMAL status
- Clear separation of the table and abnormal values section
- Concise, medically reasonable advice for abnormal values
- No hallucinated tests that are not in the report

Return ONLY a number between 0 and 10. Do not add any words.
"""
    try:
        raw, _ = gemini_generate(
            client,
            judge_prompt,
            model=model,
            temperature=0.0,
            max_output_tokens=10,
        )
        score_str = raw.strip().split()[0]
        score = float(score_str)
        if score < 0:
            score = 0.0
        if score > 10:
            score = 10.0
        return score
    except Exception:
        return 0.0



def grpo_optimize_report(
    client,
    text,
    model="gemini-2.5-flash",
    language="english",
    num_candidates=2,
):
    """
    GRPO-style selection for reports:
    1. Generate multiple candidates.
    2. Score each with a reward model.
    3. Return the highest-scoring candidate.
    """
    candidates = grpo_generate_report_candidates(
        client,
        text,
        model=model,
        language=language,
        num_candidates=num_candidates,
    )

    scored = []
    for c in candidates:
        s = grpo_score_report_candidate(
            client,
            original_text=text,
            candidate_text=c["text"],
            model=model,
        )
        scored.append({"score": s, "text": c["text"], "raw_resp": c["raw_resp"]})

    if not scored:
        return "No analysis available (GRPO scoring failed)", None

    best = max(scored, key=lambda x: x["score"])
    return best["text"], {"scored_candidates": scored}


# High-level orchestrator used by Streamlit app
def extract_and_analyze(
    uploaded_file,
    gemini_model_name="gemini-2.5-flash",
    prefer_gemini_for_ocr=True,
    language="english",
    use_grpo_for_prescription=True,
    use_grpo_for_report=True,
):
    """
    Returns dict:
      extracted_text, extractor, doc_type, analysis_text,
      analysis_text_english_for_pdf, raw_extraction_resp, raw_analysis_resp
    """
    client = init_client()
    pil_img = None
    try:
        pil_img = uploaded_file_to_pil(uploaded_file)
    except Exception:
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
            extracted_text, raw_extract_resp = gemini_extract_text_from_image(
                client, pil_img, model=gemini_model_name, instruction=instruction
            )
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
        # re-init if needed (keeps your pattern)
        client = init_client()

        if doc_type == "prescription":
            if use_grpo_for_prescription:
                # GRPO-style multi-sampling + reward-based selection
                analysis_text, analysis_resp = grpo_optimize_prescription(
                    client,
                    extracted_text,
                    model=gemini_model_name,
                    language=language,
                    num_candidates=2,
                )
            else:
                # normal single-call analysis
                analysis_text, analysis_resp = analyze_prescription_text(
                    client,
                    extracted_text,
                    model=gemini_model_name,
                    language=language,
                )

            # also produce english version for PDF
            if language.lower().startswith("en"):
                analysis_text_english = analysis_text
            else:
                # keep English version simple (no GRPO needed)
                analysis_text_english, _ = analyze_prescription_text(
                    client,
                    extracted_text,
                    model=gemini_model_name,
                    language="english",
                )

        elif doc_type == "report":
            if use_grpo_for_report:
                # GRPO-style multi-sampling + reward-based selection for reports
                analysis_text, analysis_resp = grpo_optimize_report(
                    client,
                    extracted_text,
                    model=gemini_model_name,
                    language=language,
                    num_candidates=2,
                )
            else:
                # normal single-call analysis
                analysis_text, analysis_resp = analyze_report_text(
                    client,
                    extracted_text,
                    model=gemini_model_name,
                    language=language,
                )

            # english for PDF
            if language.lower().startswith("en"):
                analysis_text_english = analysis_text
            else:
                # keep English version simple (no GRPO needed)
                analysis_text_english, _ = analyze_report_text(
                    client,
                    extracted_text,
                    model=gemini_model_name,
                    language="english",
                )

        else:
            # ask Gemini to decide and provide short analysis in requested language
            combined_prompt = f"""
Decide whether the text is a prescription or report. Then provide a concise, short analysis (bulleted) in the requested language: { 'English' if language.lower().startswith('en') else 'Kannada' }.
Text:
{extracted_text}
"""
            if client is not None:
                analysis_text, _ = gemini_generate(
                    client,
                    combined_prompt,
                    model=gemini_model_name,
                    temperature=0.0,
                    max_output_tokens=700,
                )

                # english for PDF
                combined_prompt_en = combined_prompt.replace("Kannada", "English")
                analysis_text_english, _ = gemini_generate(
                    client,
                    combined_prompt_en,
                    model=gemini_model_name,
                    temperature=0.0,
                    max_output_tokens=700,
                )
            else:
                analysis_text = "No analysis available (client not configured)"
                analysis_text_english = "No analysis available (client not configured)"
            
    except Exception as e:
        analysis_text = f"Error during analysis: {e}"
        analysis_text_english = f"Error during analysis: {e}"
    return {
        "extracted_text": extracted_text,
        "extractor": extractor_used,
        "doc_type": doc_type,
        "analysis_text": analysis_text,
        "analysis_text_english": analysis_text_english,
        "raw_extract_resp": raw_extract_resp,
        "raw_analysis_resp": analysis_resp
    }