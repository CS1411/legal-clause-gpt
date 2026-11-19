"""Lesson 5: fine-tune with the Trainer. The loss covers only the label token. About 10 minutes on a CPU."""
import sys
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, default_data_collator

HERE = Path(__file__).resolve().parent
for lesson in ("lesson2", "lesson4"):
    sys.path.insert(0, str(HERE.parent / lesson))
from ledgar import balanced_subset, ds, label_token, names  # noqa: E402
from lesson4 import extend_tokenizer_and_model  # noqa: E402

OUT = HERE / "legalgpt-ft"  # git-ignored


def make_encoder(tok, max_len=96):
    def encode(e):
        "clause + <Label>, padded to a fixed length, with every target -100 except the single label-token position."
        label = label_token(e["label"])
        enc = tok(e["text"].strip() + " " + label, truncation=True, padding="max_length", max_length=max_len)
        label_id = tok.convert_tokens_to_ids(label)
        labels = [-100] * len(enc["input_ids"])  # PyTorch's loss ignores -100, so the model is graded on one question
        if label_id in enc["input_ids"]:
            labels[enc["input_ids"].index(label_id)] = label_id
        enc["labels"] = labels
        return enc
    return encode


if __name__ == "__main__":
    torch.manual_seed(0)
    train = balanced_subset(ds["train"], per_class=100, seed=0)
    tok, model = extend_tokenizer_and_model()
    tokenized = train.map(make_encoder(tok), remove_columns=train.column_names)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(output_dir=str(OUT), num_train_epochs=3, per_device_train_batch_size=8,
                             learning_rate=5e-4, logging_steps=50, report_to="none", save_strategy="no",
                             disable_tqdm=True)
    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=default_data_collator)
    trainer.train()
    trainer.save_model(str(OUT))
    tok.save_pretrained(str(OUT))
    print(f"\nsaved fine-tuned model to {OUT.name}/ (Lessons 6-8 load it)")
