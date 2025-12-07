import os
import io
import telebot
from telebot import types
from dotenv import load_dotenv
import traceback
import tempfile
from gtts import gTTS

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT

# 👉 Import your core OCR + analysis pipeline from app.py
from app import extract_and_analyze

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # not used directly here, but app.py will read it

if not TELEGRAM_TOKEN:
    print("Error: TELEGRAM_TOKEN not found in .env")
    exit(1)

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in .env (app.py will also need this)")
    # We don't exit here so you see the error from app.py clearly if missing.

bot = telebot.TeleBot(TELEGRAM_TOKEN)

# State dictionary to store file_ids while waiting for language selection
user_states = {}

print("RxOrbit Bot started with app.py backend...")

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def text_to_speech(text, language="English"):
    print(f"DEBUG: Generating audio for language: {language}")
    try:
        lang_code = "kn" if language == "Kannada" else "en"
        tts = gTTS(text=text, lang=lang_code, slow=False)

        # Use a real temporary file instead of memory stream
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_filename = temp_audio.name
            tts.save(temp_filename)

        print(f"DEBUG: Audio generated successfully at {temp_filename}")
        return open(temp_filename, 'rb')  # Return open file object
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
                safe_line = (
                    line.replace('&', '&amp;')
                        .replace('<', '&lt;')
                        .replace('>', '&gt;')
                )
                para = Paragraph(safe_line, body_style)
                story.append(para)
                story.append(Spacer(1, 6))

        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"PDF generation error: {str(e)}")
        return None


# Small helper class to mimic Streamlit's UploadedFile object
class TelegramUpload(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name  # app.py uses .name to detect extension
    # .read() and .seek() already come from BytesIO


# ---------------------------------------------------------
# BOT HANDLERS
# ---------------------------------------------------------

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(
        message,
        "👋 Welcome to RxOrbit Bot!\n\n"
        "Send me a clear photo of a **Medical Report** or **Prescription**, and I will analyze it for you.\n\n"
        "I can provide:\n"
        "1. 📄 Detailed Text Analysis\n"
        "2. 📑 PDF Report\n"
        "3. 🔊 Audio Explanation"
    )


@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    try:
        # Save file_id in memory
        file_id = message.photo[-1].file_id
        user_states[message.chat.id] = file_id

        # Create Inline Keyboard
        markup = types.InlineKeyboardMarkup()
        btn_en = types.InlineKeyboardButton("🇬🇧 English", callback_data="English")
        btn_kn = types.InlineKeyboardButton("🇮🇳 Kannada", callback_data="Kannada")
        markup.add(btn_en, btn_kn)

        bot.reply_to(
            message,
            "Please select the language for analysis / ವಿಶ್ಲೇಷಣೆಗಾಗಿ ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ:",
            reply_markup=markup
        )

    except Exception as e:
        bot.reply_to(message, f"❌ Error: {str(e)}")


@bot.callback_query_handler(func=lambda call: call.data in ["English", "Kannada"])
def callback_lang_selection(call):
    chat_id = call.message.chat.id
    language = call.data  # "English" or "Kannada"

    # Remove the loading animation/button click effect
    bot.answer_callback_query(call.id)

    # Check if we have a pending file for this user
    if chat_id not in user_states:
        bot.send_message(chat_id, "⚠️ Session expired or no file found. Please upload the image again.")
        return

    file_id = user_states.pop(chat_id)  # Retrieve and remove from state

    # Edit the message to remove buttons and show status
    bot.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text=f"✅ Selected: {language}\n🔍 Extracting text and analyzing... Please wait."
    )

    # Process the image
    process_image(chat_id, file_id, language)


def process_image(chat_id, file_id, language):
    """
    This now uses app.extract_and_analyze(...) for:
    - OCR (Gemini or Tesseract)
    - Doc type detection
    - Prescription/report analysis
    """
    try:
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Wrap Telegram bytes into a fake "uploaded_file" for app.py
        # We give a dummy name with .jpg extension so PDF detection doesn't trigger.
        uploaded = TelegramUpload(downloaded_file, name="telegram_image.jpg")

        # Map language to what app.py expects: "english" / "kannada"
        lang_for_model = "english" if language == "English" else "kannada"

        # Use GRPO flags as False for speed in bot (you can toggle to True if you want)
        result = extract_and_analyze(
            uploaded_file=uploaded,
            language=lang_for_model,
            use_grpo_for_prescription=False,
            use_grpo_for_report=False,
        )

        doc_type = result.get("doc_type", "unknown").capitalize()
        analysis = result.get("analysis_text", "No analysis available.")
        extracted_text = result.get("extracted_text", "")

        # 3. Send Text Response
        # We'll avoid aggressive Markdown to prevent formatting issues
        response_msg = f"✅ Detected: {doc_type}\n\n{analysis}"

        if len(response_msg) > 4000:
            bot.send_message(chat_id, response_msg[:4000] + "...")
            bot.send_message(chat_id, response_msg[4000:])
        else:
            bot.send_message(chat_id, response_msg)

        # 4. Generate & Send PDF (use English version if available, otherwise the same text)
        pdf_content = analysis
        pdf_bytes = generate_pdf(pdf_content, doc_type)
        if pdf_bytes:
            bot.send_document(
                chat_id,
                pdf_bytes,
                visible_file_name=f"{doc_type}_Analysis.pdf",
                caption="Here is your PDF report 📑"
            )

        # 5. Generate & Send Audio
        audio_bytes = text_to_speech(analysis, language)
        if audio_bytes:
            bot.send_audio(
                chat_id,
                audio_bytes,
                title=f"{doc_type} Analysis",
                caption="Listen to the analysis 🔊"
            )
            if hasattr(audio_bytes, 'close'):
                audio_bytes.close()
        else:
            bot.send_message(chat_id, "⚠️ Audio generation failed. Please check server logs.")

    except Exception as e:
        bot.send_message(chat_id, f"❌ An error occurred: {str(e)}")
        print(traceback.format_exc())


@bot.message_handler(content_types=['document'])
def handle_docs_pdf(message):
    bot.reply_to(
        message,
        "Currently I mostly support images (JPG/PNG). "
        "If you sent a PDF, please convert it to an image or screenshot it for best results."
    )


if __name__ == "__main__":
    bot.infinity_polling()
