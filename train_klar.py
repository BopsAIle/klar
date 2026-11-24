import os
import json
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from klar.model.klar_net import KLaRLightningModule


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train & Evaluate KLaR (Spec-Compliant)")
    parser.add_argument("--pro_data_dir", default="./data/kegg50k/pro_data", help="Path to pro_data directory")
    parser.add_argument(
        "--encoder",
        default="pubmedbert",
        choices=["pubmedbert", "bert", "flant5", "qwen3"],
        help="Which precomputed frozen textual encoder embedding to load",
    )
    parser.add_argument("--d_s", type=int, default=256, help="Structural RGNN hidden dimension")
    parser.add_argument("--d", type=int, default=256, help="Unified fusion & MoE hidden dimension")
    parser.add_argument("--num_rgnn_layers", type=int, default=2, help="Number of RGNN layers L")
    parser.add_argument("--num_experts", type=int, default=64, help="Number of MoE experts N")
    parser.add_argument("--top_k", type=int, default=4, help="Top-K routed experts")
    parser.add_argument("--num_negatives", type=int, default=128, help="Negative samples per positive triple")
    parser.add_argument("--margin", type=float, default=1.0, help="Max-margin ranking loss margin gamma")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature tau")
    parser.add_argument("--lambda_align", type=float, default=0.1, help="Weight lambda_1 for L_align")
    parser.add_argument("--lambda_moe", type=float, default=0.01, help="Weight lambda_2 for L_moe")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size (optimized for RTX 5090)")
    parser.add_argument("--max_epochs", type=int, default=30, help="Max epochs per seed")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds to run (e.g. 42 100 2026)")
    parser.add_argument("--output_dir", default="./output", help="Directory to store checkpoints and results")

    # Ablation flags
    parser.add_argument("--wo_kge", action="store_true", help="Disable Structural RGNN branch")
    parser.add_argument("--wo_texte", action="store_true", help="Disable Textual branch")
    parser.add_argument("--wo_align", action="store_true", help="Disable InfoNCE alignment loss")
    parser.add_argument("--wo_moe", action="store_true", help="Disable MoE (use 1 expert)")
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model: KLaRLightningModule, loader: DataLoader, device: str) -> dict:
    model.eval()
    z_all, _, _ = model.compute_entity_representations()
    all_ranks = []
    for batch in loader:
        batch = batch.to(device)
        ranks = model._evaluate_filtered_batch(batch, z_all)
        all_ranks.extend(ranks)

    ranks_t = torch.tensor(all_ranks, dtype=torch.float32)
    return {
        "mrr": float((1.0 / ranks_t).mean().item()),
        "hits1": float((ranks_t <= 1).float().mean().item()),
        "hits3": float((ranks_t <= 3).float().mean().item()),
        "hits10": float((ranks_t <= 10).float().mean().item()),
    }


def main():
    args = parse_args()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_pt = torch.load(os.path.join(args.pro_data_dir, "train_triples.pt"))
    valid_pt = torch.load(os.path.join(args.pro_data_dir, "valid_triples.pt"))
    test_pt = torch.load(os.path.join(args.pro_data_dir, "test_triples.pt"))
    all_true_pt = torch.load(os.path.join(args.pro_data_dir, "all_true_triples.pt"))

    emb_path = os.path.join(args.pro_data_dir, f"text_emb_{args.encoder.lower()}.pt")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"Precomputed textual embedding not found at {emb_path}. "
            f"Run: python klar/data/precompute_text_emb.py --pro_data_dir {args.pro_data_dir} --encoder {args.encoder}"
        )
    text_emb = torch.load(emb_path)

    with open(os.path.join(args.pro_data_dir, "entity2id.txt"), "r", encoding="utf-8") as f:
        num_entities = sum(1 for line in f if line.strip())
    with open(os.path.join(args.pro_data_dir, "relation2id.txt"), "r", encoding="utf-8") as f:
        num_relations = sum(1 for line in f if line.strip())

    print(
        f"Dataset loaded on {device}: |V|={num_entities}, |R|={num_relations}, "
        f"train={len(train_pt)}, valid={len(valid_pt)}, test={len(test_pt)}, "
        f"text_emb={tuple(text_emb.shape)}"
    )

    train_loader = DataLoader(train_pt, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(valid_pt, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_pt, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    os.makedirs(args.output_dir, exist_ok=True)
    all_seed_metrics = []

    for seed in args.seeds:
        print(f"\n==================== RUNNING SEED {seed} (Encoder: {args.encoder}) ====================")
        seed_everything(seed)

        exp_name = f"KLaR_{args.encoder}_E{args.num_experts}_K{args.top_k}_seed{seed}"
        ckpt_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        best_ckpt_path = os.path.join(ckpt_dir, "best_model.pt")

        model = KLaRLightningModule(
            num_entities=num_entities,
            num_relations=num_relations,
            text_embeddings=text_emb,
            train_triples=train_pt,
            all_true_triples=all_true_pt,
            d_s=args.d_s,
            d=args.d,
            num_rgnn_layers=args.num_rgnn_layers,
            num_experts=args.num_experts,
            top_k=args.top_k,
            num_negatives=args.num_negatives,
            margin=args.margin,
            temperature=args.temperature,
            lambda_align=args.lambda_align,
            lambda_moe=args.lambda_moe,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            wo_kge=args.wo_kge,
            wo_texte=args.wo_texte,
            wo_align=args.wo_align,
            wo_moe=args.wo_moe,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-5)

        best_val_mrr = -1.0
        patience_counter = 0

        for epoch in range(1, args.max_epochs + 1):
            model.train()
            epoch_losses = []
            for batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.max_epochs:02d}", leave=False):
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = model.training_step(batch, 0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            scheduler.step()
            train_loss = float(np.mean(epoch_losses))
            val_metrics = evaluate_model(model, val_loader, device)

            print(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
                f"Val MRR: {val_metrics['mrr']:.4f} | H@1: {val_metrics['hits1']:.4f} | "
                f"H@3: {val_metrics['hits3']:.4f} | H@10: {val_metrics['hits10']:.4f}"
            )

            if val_metrics["mrr"] > best_val_mrr:
                best_val_mrr = val_metrics["mrr"]
                patience_counter = 0
                torch.save(model.state_dict(), best_ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch} (Best Val MRR: {best_val_mrr:.4f})")
                    break

        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        test_metrics = evaluate_model(model, test_loader, device)
        print(
            f"[Seed {seed} Test Results] MRR: {test_metrics['mrr']:.4f} | "
            f"H@1: {test_metrics['hits1']:.4f} | H@3: {test_metrics['hits3']:.4f} | "
            f"H@10: {test_metrics['hits10']:.4f}"
        )
        all_seed_metrics.append(
            {
                "Test-mrr": test_metrics["mrr"],
                "Test-hits1": test_metrics["hits1"],
                "Test-hits3": test_metrics["hits3"],
                "Test-hits10": test_metrics["hits10"],
            }
        )

    if all_seed_metrics:
        summary = {}
        print("\n==================== FINAL SUMMARY ACROSS SEEDS ====================")
        for k in ["Test-mrr", "Test-hits1", "Test-hits3", "Test-hits10"]:
            vals = [m[k] for m in all_seed_metrics if k in m]
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            summary[k] = {"mean": mean_v, "std": std_v, "per_seed": vals}
            print(f"{k:15s}: {mean_v:.4f} ± {std_v:.4f}")

        summary_path = os.path.join(args.output_dir, f"results_kegg50k_{args.encoder}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary metrics to {summary_path}")


if __name__ == "__main__":
    main()
