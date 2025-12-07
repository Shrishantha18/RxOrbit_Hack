import os
import io
import telebot
from telebot import types
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import traceback
import tempfile
from gtts import gTTS
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT

import time
import random

# ... (previous imports)

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TELEGRAM_TOKEN:
    print("Error: TELEGRAM_TOKEN not found in .env")
    exit(1)

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)
bot = telebot.TeleBot(TELEGRAM_TOKEN)
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# State dictionary to store file_ids while waiting for language selection
user_states = {}

print("RxOrbit Bot started...")

def retry_gemini_call(func, *args, **kwargs):
    """Retries a Gemini API call with exponential backoff."""
    max_retries = 3
    base_delay = 5  # Start with 5 seconds
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit hit. Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise e
    raise Exception("Max retries exceeded for Gemini API call.")

# ---------------------------------------------------------
# HELPER FUNCTIONS (Lifted from streamlit_app1.py)
# ---------------------------------------------------------

def gemini_extract_text(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        prompt = """
        Extract ALL text from this medical document clearly.
        Maintain line breaks. Do NOT summarize.
        """
        response = retry_gemini_call(model.generate_content, [prompt, image])
        return response.text.strip()
    except Exception as e:
        return f"OCR error: {str(e)}"

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
        response = retry_gemini_call(model.generate_content, prompt)
        return response.text
    except Exception as e:
        return f"Report analysis failed: {str(e)}"

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
        response = retry_gemini_call(model.generate_content, prompt)
        return response.text
    except Exception as e:
        return f"Prescription analysis failed: {str(e)}"

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

def text_to_speech(text, language="English"):
    print(f"DEBUG: Generating audio for language: {language}")
    try:
        lang_code = "kn" if language == "Kannada" else "en"
        tts = gTTS(text=text, lang=lang_code, slow=False)
        
        # Use a real temporary file instead of memory stream
        # delete=False is important so we can re-open it to send
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_filename = temp_audio.name
            tts.save(temp_filename)
            
        print(f"DEBUG: Audio generated successfully at {temp_filename}")
        return open(temp_filename, 'rb') # Return open file object
    except Exception as e:
        print(f"Voice generation error: {str(e)}")
        traceback.print_exc()
        return None

def generate_pdf(content, doc_type):
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor='darkblue',
            spaceAfter=30,
            alignment=TA_LEFT
        )
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            leading=16,
            alignment=TA_LEFT
        )
        
        title = Paragraph(f"Medical Analysis: {doc_type}", title_style)
        story.append(title)
        story.append(Spacer(1, 12))
        
        for line in content.split('\n'):
            if line.strip():
                safe_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                para = Paragraph(safe_line, body_style)
                story.append(para)
                story.append(Spacer(1, 6))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"PDF generation error: {str(e)}")
        return None

# ---------------------------------------------------------
# BOT HANDLERS
# ---------------------------------------------------------

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "👋 Welcome to RxOrbit Bot!\n\nSend me a clear photo of a **Medical Report** or **Prescription**, and I will analyze it for you.\n\nI can provide:\n1. 📄 Detailed Text Analysis\n2. 📑 PDF Report\n3. 🔊 Audio Explanation")

@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    try:
        # Save file_id in memory
        file_id = message.photo[-1].file_id
        user_states[message.chat.id] = file_id
        
        # Create Inline Keyboard
        markup = types.InlineKeyboardMarkup()
        btn_en = types.InlineKeyboardButton("English", callback_data="English")
        btn_kn = types.InlineKeyboardButton("Kannada", callback_data="Kannada")
        markup.add(btn_en, btn_kn)
        
        bot.reply_to(message, "Please select the language for analysis / ವಿಶ್ಲೇಷಣೆಗಾಗಿ ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ:", reply_markup=markup)
        
    except Exception as e:
        bot.reply_to(message, f"❌ Error: {str(e)}")

@bot.callback_query_handler(func=lambda call: call.data in ["English", "Kannada"])
def callback_lang_selection(call):
    chat_id = call.message.chat.id
    language = call.data
    
    # Remove the loading animation/button click effect
    bot.answer_callback_query(call.id)
    
    # Check if we have a pending file for this user
    if chat_id not in user_states:
        bot.send_message(chat_id, "⚠️ Session expired or no file found. Please upload the image again.")
        return

    file_id = user_states.pop(chat_id) # Retrieve and remove from state
    
    # Edit the message to remove buttons and show status
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, 
                          text=f"✅ Selected: {language}\n🔍 Extracting text and analyzing... Please wait.")
    
    # Process the image
    process_image(chat_id, file_id, language)

def process_image(chat_id, file_id, language):
    try:
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # 1. Extract Text
        extracted_text = gemini_extract_text(downloaded_file)
        if extracted_text.startswith("OCR error"):
            bot.send_message(chat_id, "Oops some error occured")
            return

        # 2. Analyze
        doc_type, analysis = auto_analyze(extracted_text, language)

        # 3. Send Text Response
        response_msg = f"✅ **Detected:** {doc_type}\n\n{analysis}"
        # Telegram limit check
        if len(response_msg) > 4000:
            bot.send_message(chat_id, response_msg[:4000] + "...")
            bot.send_message(chat_id, response_msg[4000:], parse_mode="Markdown")
        else:
            bot.send_message(chat_id, response_msg, parse_mode="Markdown")

        # 4. Generate & Send PDF
        pdf_bytes = generate_pdf(analysis, doc_type)
        if pdf_bytes:
            bot.send_document(chat_id, pdf_bytes, visible_file_name=f"{doc_type}_Analysis.pdf", caption="Here is your PDF report 📑")

        # 5. Generate & Send Audio
        audio_bytes = text_to_speech(analysis, language)
        if audio_bytes:
            bot.send_audio(chat_id, audio_bytes, title=f"{doc_type} Analysis", caption="Listen to the analysis 🔊")
            # Close the file if it's an open file object
            if hasattr(audio_bytes, 'close'):
                audio_bytes.close()
        else:
             bot.send_message(chat_id, "⚠️ Audio generation failed. Please check server logs.")

    except Exception as e:
        bot.send_message(chat_id, f"❌ An error occurred: {str(e)}")
        print(traceback.format_exc())

@bot.message_handler(content_types=['document'])
def handle_docs_pdf(message):
    bot.reply_to(message, "Currently I mostly support images (JPG/PNG). If you sent a PDF, please convert it to an image or screenshot it for best results.")

if __name__ == "__main__":
    bot.infinity_polling()

