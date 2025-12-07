"""
finetune_donut.py

Fine-tunes naver-clova-ix/donut-base for image -> structured text (JSON-like).

Expected inputs:
 - data/finetune_labels.jsonl  (lines of {"image": "<relative path>", "target":"<string target>"})
 - image files referenced in finetune_labels.jsonl (e.g., data/prescriptions_processed/*.png and data/processed_images/*.png)

Outputs:
 - Saved model/tokenizer/processor in ./finetuned_donut
Note: GPU is strongly recommended.
"""

import os
import json
from pathlib import Path

from datasets import Dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
import torch
from PIL import Image

# ---------- CONFIG ----------
MODEL_CHECKPOINT = "naver-clova-ix/donut-base"
FINETUNE_FILE = "./data/finetune_labels.jsonl"   # JSONL mapping image -> target string
OUTPUT_DIR = "./finetuned_donut"
EPOCHS = 1   # keep epochs low to shorten run time/steps
BATCH_SIZE = 1   # reduce if OOM on GPU
LEARNING_RATE = 5e-5
MAX_TARGET_LENGTH = 256
IMG_SIZE = 512   # reduce to ease memory pressure
# ----------------------------


def main():
    # Load Donut processor and model
    print("Loading Donut processor & model:", MODEL_CHECKPOINT)
    processor = DonutProcessor.from_pretrained(MODEL_CHECKPOINT)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_CHECKPOINT)
    # Ensure decoder start and pad tokens are set for training
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Read finetune_labels.jsonl
    if not os.path.exists(FINETUNE_FILE):
        raise SystemExit(
            f"Finetune label file not found: {FINETUNE_FILE}\n"
            'Create a JSONL file where each line is: {"image":"data/prescriptions_processed/xxx.png", "target": "{...}"}'
        )

    examples = []
    with open(FINETUNE_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line.strip())
            img_path = obj["image"]
            if not os.path.exists(img_path):
                raise SystemExit(f"Image not found: {img_path}")
            examples.append({"image": img_path, "target": obj["target"]})

    print(f"Loaded {len(examples)} examples")

    # Build a datasets.Dataset
    ds = Dataset.from_list(examples)

    # Preprocess (batched) to avoid huge memory spikes
    def preprocess_fn(batch):
        images = [Image.open(p).convert("RGB") for p in batch["image"]]
        inputs = processor(images, return_tensors="pt", size=IMG_SIZE)
        with processor.tokenizer.as_target_tokenizer():
            label_ids = processor.tokenizer(
                batch["target"],
                max_length=MAX_TARGET_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).input_ids
        # Return lists of arrays for datasets
        return {
            "pixel_values": [pv.numpy() for pv in inputs.pixel_values],
            "labels": [l.numpy() for l in label_ids],
        }

    print("Preprocessing dataset (this may take a moment)...")
    ds_proc = ds.map(
        preprocess_fn,
        remove_columns=ds.column_names,
        batched=True,
        batch_size=2,
        num_proc=1,  # avoid multiprocessing issues on Windows
    )

    class TorchDataset(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            item = self.ds[idx]
            return {
                "pixel_values": torch.tensor(item["pixel_values"]),
                "labels": torch.tensor(item["labels"]),
            }

    train_dataset = TorchDataset(ds_proc)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        do_train=True,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        tokenizer=processor.tokenizer,
    )

    print("Starting training on device:", device)
    trainer.train()
    print("Training finished. Saving model to:", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Saved finetuned processor+model at", OUTPUT_DIR)


if __name__ == "__main__":
    main()


