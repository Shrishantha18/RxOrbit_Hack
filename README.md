# Medical Document Analyzer with Fine-tuned Donut Model

An advanced application that leverages a fine-tuned Donut model for accurate extraction and analysis of medical documents, with support for multiple languages including English and Kannada.

## Features

- **Document Analysis**: Automatically extracts and analyzes medical reports and prescriptions
- **Fine-tuned Donut Model**: Specialized OCR model trained specifically for medical documents
- **Multilingual Support**: Provides analysis in both English and Kannada
- **Voice Output**: Converts analysis to speech for better accessibility
- **Export Options**: Save analysis as text or PDF

## Prerequisites

- Python 3.8+
- pip
- Tesseract OCR (for fallback text extraction)
- Poppler (for PDF processing)
- CUDA-compatible GPU (recommended for better performance)

## Installation

1. Clone the repository:
   ```bash
   git clone [your-repository-url]
   cd Devofolio_Hackathon
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. Install PyTorch with CUDA support (recommended):
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up environment variables (if needed):
   - Create a `.env` file in the project root
   - Add any required API keys or configurations

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Open the app in your browser (usually http://localhost:8501)

3. Upload a medical document (PNG, JPG, or PDF)

4. Select your preferred output language (English or Kannada)

5. View the analysis and use the available options:
   - Listen to the audio version
   - Download as text file
   - Download as PDF

## Model Details

The application uses a fine-tuned Donut model that has been specifically trained on medical documents to achieve high accuracy in:
- Text extraction from various medical document formats
- Structured data extraction from prescriptions and lab reports
- Handling of medical terminology and abbreviations

## Project Structure

```
Devofolio_Hackathon/
├── .env                    # Environment variables
├── finetune_donut.py          # Core application logic
├── streamlit_app.py       # Streamlit UI and application flow
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── test_donut.py         # Script to test the Donut model
└── data/                 # (Optional) Sample data directory
```

## Dependencies

- streamlit
- torch
- transformers
- python-dotenv
- gTTS
- reportlab
- Pillow
- pytesseract
- pdf2image
- numpy

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for the Donut model architecture
- [NAVER CLOVA](https://github.com/clovaai/donut) for the original Donut implementation
- [Streamlit](https://streamlit.io/) for the web interface
