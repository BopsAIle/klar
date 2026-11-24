import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, T5EncoderModel

ENCODER_REGISTRY = {
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "bert": "bert-base-uncased",
    "flant5": "google/flan-t5-base",
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
}


def load_frozen_encoder(encoder_key: str, model_name_override: str = None, device: str = "cuda"):
    model_name = model_name_override or ENCODER_REGISTRY.get(encoder_key.lower(), encoder_key)
    print(f"Loading frozen textual encoder: {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "t5" in model_name.lower():
        encoder = T5EncoderModel.from_pretrained(model_name)
    else:
        encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.to(device)
    return tokenizer, encoder


@torch.no_grad()
def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def main():
    parser = argparse.ArgumentParser(description="Pre-compute frozen sentence embeddings t_v for KLaR")
    parser.add_argument("--pro_data_dir", required=True, help="Path to dataset pro_data directory")
    parser.add_argument(
        "--encoder",
        default="pubmedbert",
        choices=["pubmedbert", "bert", "flant5", "qwen3"],
        help="Which frozen textual encoder to use",
    )
    parser.add_argument("--model_name", default=None, help="Optional custom HuggingFace model ID")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max token sequence length |X_v|")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    corpus_path = os.path.join(args.pro_data_dir, "entity_corpus.jsonl")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}. Run dataset preprocessing first.")

    records = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    records.sort(key=lambda r: int(r["entity_id"]))
    texts = [r["text"] for r in records]
    print(f"Loaded {len(texts)} entity contexts from {corpus_path}")

    tokenizer, encoder = load_frozen_encoder(args.encoder, args.model_name, args.device)

    all_embeddings = []
    for start_idx in tqdm(range(0, len(texts), args.batch_size), desc=f"Encoding ({args.encoder})"):
        batch_texts = texts[start_idx : start_idx + args.batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(args.device)
        attention_mask = encoded["attention_mask"].to(args.device)

        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        all_embeddings.append(pooled.cpu().float())

    text_emb_matrix = torch.cat(all_embeddings, dim=0)
    out_path = os.path.join(args.pro_data_dir, f"text_emb_{args.encoder.lower()}.pt")
    torch.save(text_emb_matrix, out_path)
    print(f"Saved precomputed textual representation matrix {tuple(text_emb_matrix.shape)} to {out_path}")


if __name__ == "__main__":
    main()
