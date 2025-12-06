"""
Test script for the fine-tuned Donut model
"""
import os
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

MODEL_PATH = "./finetuned_donut"

if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}")
    exit(1)

print("Loading fine-tuned Donut model...")
processor = DonutProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}")

# Test with a sample image
test_image_path = input("Enter path to test image (or press Enter to skip): ").strip()
if test_image_path and os.path.exists(test_image_path):
    print(f"\nProcessing {test_image_path}...")
    image = Image.open(test_image_path).convert("RGB")
    
    # Process image
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    # Generate
    decoder_input_ids = processor.tokenizer(
        "<s>", add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    
    # Decode
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = sequence.replace("<s>", "").replace("</s>", "").strip()
    
    print("\nExtracted text:")
    print(sequence)
else:
    print("\nModel loaded successfully! Ready to use in Streamlit app.")

