"""Lesson 4: teach the tokenizer new words. Add one token per clause type, resize the embeddings, set a pad token."""
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson2"))
from ledgar import TOP, label_token  # noqa: E402


def extend_tokenizer_and_model(base="distilgpt2"):
    """The three steps: add the label tokens, resize the model's embeddings to match, give GPT-2 a pad token.

    mean_resizing=False: transformers 5 would otherwise initialise every new row to the same mean vector, which
    makes the six label tokens identical to the model; this keeps them distinct random rows as in earlier versions.
    """
    tok = AutoTokenizer.from_pretrained(base)
    tok.add_tokens([label_token(i) for i in TOP])  # 1. one atomic token per clause type
    tok.pad_token = tok.eos_token                  # 3. GPT-2 ships without a pad token
    model = AutoModelForCausalLM.from_pretrained(base)
    model.resize_token_embeddings(len(tok), mean_resizing=False)  # 2. one new embedding row per new token
    return tok, model


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # GPT-2 marks a leading space with 'Ġ', which some consoles cannot print
    plain = AutoTokenizer.from_pretrained("distilgpt2")
    demo = label_token(TOP[0])
    before = plain.tokenize(demo)
    print(f"BEFORE: {demo!r} -> {len(before)} tokens:", before)

    tok, model = extend_tokenizer_and_model()
    after = tok.tokenize(demo)
    print(f"AFTER : {demo!r} -> {len(after)} token :", after)
    print(f"\nembeddings resized: 50257 -> {model.get_input_embeddings().weight.shape[0]} rows")

    print("\nBPE can spell any word, so no vocabulary words are added, only the six clause types:")
    for w in ["indemnification", "hereinafter", "notwithstanding", "Delaware", "zzxqwv"]:
        print(f"  {w:16s} -> {tok.tokenize(w)}")

    batch = ["This Agreement shall be governed by the laws of the State of Delaware.",
             "Confidential Information stays secret."]
    enc = tok(batch, padding=True, return_tensors="pt")
    print("\nbatch shape (rows x tokens):", tuple(enc["input_ids"].shape))
    for i in range(2):
        print(f"  clause {i}: mask {enc['attention_mask'][i].tolist()}  (0 = padding, ignored)")
    print(f"pad token = {tok.pad_token!r} (id {tok.pad_token_id}), reused from end-of-text")

    assert len(before) == 6 and len(after) == 1
    assert model.get_input_embeddings().weight.shape[0] == len(tok) == 50263
