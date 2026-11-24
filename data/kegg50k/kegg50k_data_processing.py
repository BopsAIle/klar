import os
import json
import pickle
import random
import argparse
import urllib.request
from collections import defaultdict
import torch
from tqdm import tqdm

KEGG_ENDPOINTS = {
    "hsa_genes.tsv": "https://rest.kegg.jp/list/hsa",
    "drug.tsv": "https://rest.kegg.jp/list/drug",
    "disease.tsv": "https://rest.kegg.jp/list/disease",
    "pathway_hsa.tsv": "https://rest.kegg.jp/list/pathway/hsa",
    "network.tsv": "https://rest.kegg.jp/list/network",
}

RELATION_TEMPLATES = {
    "PATHWAY_GENE": {
        "forward": "involves gene",
        "backward": "participates in pathway",
    },
    "DRUG_TARGET_GENE": {
        "forward": "targets gene",
        "backward": "is targeted by drug",
    },
    "NETWORK_GENE": {
        "forward": "includes gene",
        "backward": "belongs to biological network",
    },
    "DISEASE_GENE": {
        "forward": "is associated with gene",
        "backward": "is associated with disease",
    },
    "DRUG_EFFICACY_DISEASE": {
        "forward": "treats disease",
        "backward": "is treated by drug",
    },
    "DISEASE_PATHWAY": {
        "forward": "alters pathway",
        "backward": "is altered in disease",
    },
    "DRUG_TARGET_PATHWAY": {
        "forward": "modulates pathway",
        "backward": "is modulated by drug",
    },
    "NETWORK_PATHWAY": {
        "forward": "is part of pathway",
        "backward": "contains biological network",
    },
    "DISEASE_NETWORK": {
        "forward": "disrupts biological network",
        "backward": "is disrupted in disease",
    },
}


def download_kegg_lists(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    for fname, url in KEGG_ENDPOINTS.items():
        fpath = os.path.join(cache_dir, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
            continue
        print(f"Fetching {url} -> {fpath} ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"Warning: could not fetch {url}: {e}")


def parse_kegg_lists(cache_dir: str) -> dict:
    id2name = {}

    # 1. Genes: hsa:10458\tCDS\t17:...\tBAIAP2, IRSp53, ...; BAR/IMD domain containing adaptor protein 2
    gene_file = os.path.join(cache_dir, "hsa_genes.tsv")
    if os.path.exists(gene_file):
        with open(gene_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    raw_id = parts[0].strip()  # e.g. hsa:10458
                    desc = parts[-1].strip()
                    if raw_id.lower().startswith("hsa:"):
                        num = raw_id.split(":", 1)[1]
                        key = f"HSA:{num}"
                        # Take primary symbol and full name
                        if ";" in desc:
                            symbols, full_name = desc.split(";", 1)
                            primary_sym = symbols.split(",")[0].strip()
                            clean_name = f"{primary_sym} ({full_name.strip()})"
                        else:
                            clean_name = desc.split(",")[0].strip()
                        id2name[key] = clean_name

    # 2. Drugs: dr:D00001 or D00001\tWater (JP18/USP); ...
    drug_file = os.path.join(cache_dir, "drug.tsv")
    if os.path.exists(drug_file):
        with open(drug_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    raw_id = parts[0].replace("dr:", "").strip()
                    desc = parts[1].strip()
                    primary_name = desc.split(";")[0].strip()
                    # Strip pharmacopoeia suffix like (JP18/USP) if desired, or keep clean
                    id2name[raw_id] = primary_name

    # 3. Diseases: ds:H00001 or H00001\tB-cell acute lymphoblastic leukemia; ...
    disease_file = os.path.join(cache_dir, "disease.tsv")
    if os.path.exists(disease_file):
        with open(disease_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    raw_id = parts[0].replace("ds:", "").strip()
                    desc = parts[1].strip()
                    primary_name = desc.split(";")[0].strip()
                    id2name[raw_id] = primary_name

    # 4. Pathways: path:hsa00010 or hsa00010\tGlycolysis / Gluconeogenesis - Homo sapiens (human)
    pathway_file = os.path.join(cache_dir, "pathway_hsa.tsv")
    if os.path.exists(pathway_file):
        with open(pathway_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    raw_id = parts[0].replace("path:", "").strip()
                    desc = parts[1].replace(" - Homo sapiens (human)", "").strip()
                    id2name[raw_id] = f"{desc} pathway" if not desc.lower().endswith("pathway") else desc

    # 5. Networks: ne:N00001 or N00001\tEGF-EGFR-RAS-ERK signaling pathway
    network_file = os.path.join(cache_dir, "network.tsv")
    if os.path.exists(network_file):
        with open(network_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    raw_id = parts[0].replace("ne:", "").strip()
                    desc = parts[1].strip()
                    id2name[raw_id] = desc

    return id2name


def fallback_entity_name(ent_id: str) -> str:
    if ent_id.startswith("HSA:"):
        return f"Human Gene {ent_id.split(':', 1)[1]}"
    elif ent_id.startswith("hsa"):
        return f"KEGG Pathway {ent_id}"
    elif ent_id.startswith("D"):
        return f"KEGG Drug {ent_id}"
    elif ent_id.startswith("H"):
        return f"KEGG Disease {ent_id}"
    elif ent_id.startswith("N"):
        return f"KEGG Network {ent_id}"
    return ent_id


def load_triples(txt_path: str, entity2id: dict, relation2id: dict):
    triples = []
    raw_triples = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split("\t")
            raw_triples.append((h, r, t))
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return raw_triples, torch.tensor(triples, dtype=torch.long)


def sample_random_walks_for_entity(
    start_node: int,
    adj_list: dict,
    num_walks: int,
    max_walk_len: int,
    id2name: dict,
    id2rel: dict,
    rng: random.Random,
) -> str:
    neighbors = adj_list.get(start_node, [])
    start_name = id2name[start_node]
    if not neighbors:
        return start_name

    walk_sentences = []
    for _ in range(num_walks):
        curr = start_node
        prev = -1
        tokens = [id2name[curr]]
        for _step in range(max_walk_len):
            cand = adj_list.get(curr, [])
            if not cand:
                break
            # Prefer non-backtracking step if multiple neighbors exist
            if len(cand) > 1 and prev != -1:
                filtered = [c for c in cand if c[0] != prev]
                next_node, rel_id, is_inv = rng.choice(filtered if filtered else cand)
            else:
                next_node, rel_id, is_inv = rng.choice(cand)
            rel_name = id2rel[rel_id]
            tmpl = RELATION_TEMPLATES.get(rel_name, {"forward": rel_name.lower(), "backward": f"reverse {rel_name.lower()}"})
            rel_phrase = tmpl["backward"] if is_inv else tmpl["forward"]
            tokens.append(rel_phrase)
            tokens.append(id2name[next_node])
            prev = curr
            curr = next_node
        walk_sentences.append(" ".join(tokens))

    # Deduplicate while preserving order
    unique_sentences = list(dict.fromkeys(walk_sentences))
    return " . ".join(unique_sentences)


def main():
    parser = argparse.ArgumentParser(description="Preprocess kegg50k for KLaR (Zero Data Leakage)")
    parser.add_argument("--data_root", default=os.path.dirname(os.path.abspath(__file__)), help="Path to data/kegg50k")
    parser.add_argument("--num_walks", type=int, default=16, help="Number of random walks M per entity")
    parser.add_argument("--max_walk_len", type=int, default=6, help="Max walk length T")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_root, "raw_data")
    pro_dir = os.path.join(args.data_root, "pro_data")
    cache_dir = os.path.join(raw_dir, "kegg_api_cache")
    os.makedirs(pro_dir, exist_ok=True)

    # Step 1: Load official entity2index and relation2index from raw_data
    with open(os.path.join(raw_dir, "entity2index.pkl"), "rb") as f:
        entity2id = pickle.load(f)
    with open(os.path.join(raw_dir, "relation2index.pkl"), "rb") as f:
        raw_rel2id = pickle.load(f)

    relation2id = {}
    for k, v in raw_rel2id.items():
        rel_name = k[1] if isinstance(k, tuple) else str(k)
        relation2id[rel_name] = int(v)

    id2entity = {v: k for k, v in entity2id.items()}
    id2rel = {v: k for k, v in relation2id.items()}

    print(f"Loaded |V| = {len(entity2id)} entities, |R| = {len(relation2id)} relations.")

    # Step 2: Fetch and resolve biological names from KEGG REST API
    download_kegg_lists(cache_dir)
    kegg_id2name = parse_kegg_lists(cache_dir)

    entity2name = {}
    id2name_int = {}
    resolved_count = 0
    for ent_str, ent_idx in entity2id.items():
        if ent_str in kegg_id2name:
            entity2name[ent_str] = kegg_id2name[ent_str]
            resolved_count += 1
        else:
            entity2name[ent_str] = fallback_entity_name(ent_str)
        id2name_int[ent_idx] = entity2name[ent_str]

    print(f"Resolved biological names via KEGG REST API: {resolved_count}/{len(entity2id)} ({resolved_count / len(entity2id) * 100:.2f}%)")

    # Save entity2id.txt, relation2id.txt, entity2name.json, relation2text.json
    with open(os.path.join(pro_dir, "entity2id.txt"), "w", encoding="utf-8") as f:
        for ent_idx in range(len(id2entity)):
            f.write(f"{id2entity[ent_idx]}\t{ent_idx}\n")

    with open(os.path.join(pro_dir, "relation2id.txt"), "w", encoding="utf-8") as f:
        for rel_idx in range(len(id2rel)):
            f.write(f"{id2rel[rel_idx]}\t{rel_idx}\n")

    with open(os.path.join(pro_dir, "entity2name.json"), "w", encoding="utf-8") as f:
        json.dump(entity2name, f, ensure_ascii=False, indent=2)

    with open(os.path.join(pro_dir, "relation2text.json"), "w", encoding="utf-8") as f:
        json.dump(RELATION_TEMPLATES, f, ensure_ascii=False, indent=2)

    # Also save legacy relation2text.txt (flat forward map) for backward compatibility
    flat_rel2text = {k: v["forward"] for k, v in RELATION_TEMPLATES.items()}
    with open(os.path.join(pro_dir, "relation2text.txt"), "w", encoding="utf-8") as f:
        json.dump(flat_rel2text, f, ensure_ascii=False)

    # Step 3: Load train / valid / test triples
    raw_train, train_pt = load_triples(os.path.join(raw_dir, "train.txt"), entity2id, relation2id)
    raw_valid, valid_pt = load_triples(os.path.join(raw_dir, "valid.txt"), entity2id, relation2id)
    raw_test, test_pt = load_triples(os.path.join(raw_dir, "test.txt"), entity2id, relation2id)

    torch.save(train_pt, os.path.join(pro_dir, "train_triples.pt"))
    torch.save(valid_pt, os.path.join(pro_dir, "valid_triples.pt"))
    torch.save(test_pt, os.path.join(pro_dir, "test_triples.pt"))

    all_true_pt = torch.cat([train_pt, valid_pt, test_pt], dim=0)
    torch.save(all_true_pt, os.path.join(pro_dir, "all_true_triples.pt"))

    # Copy train.txt, val.txt, test.txt into pro_data
    for split_name, raw_list in [("train", raw_train), ("val", raw_valid), ("test", raw_test)]:
        with open(os.path.join(pro_dir, f"{split_name}.txt"), "w", encoding="utf-8") as f:
            for h, r, t in raw_list:
                f.write(f"{h}\t{r}\t{t}\n")

    # Step 4: Build adjacency graph STRICTLY from train triples (Zero Data Leakage!)
    train_set = set(map(tuple, train_pt.tolist()))
    test_set = set(map(tuple, test_pt.tolist()))
    leakage = train_set.intersection(test_set)
    print(f"Train triples: {len(train_pt)}, Valid triples: {len(valid_pt)}, Test triples: {len(test_pt)}")
    print(f"Exact overlap between train and test splits: {len(leakage)}")

    adj_list = defaultdict(list)
    reverse_graph = {}
    for h_str, r_str, t_str in raw_train:
        h_id = entity2id[h_str]
        r_id = relation2id[r_str]
        t_id = entity2id[t_str]
        adj_list[h_id].append((t_id, r_id, False))
        adj_list[t_id].append((h_id, r_id, True))

        # Populate legacy reverse_graph using ONLY train edges
        if h_str not in reverse_graph:
            reverse_graph[h_str] = {}
        if t_str not in reverse_graph:
            reverse_graph[t_str] = {}
        reverse_graph[h_str][t_str] = r_str
        reverse_graph[t_str][h_str] = r_str

    with open(os.path.join(pro_dir, "reverse_graph.json"), "w", encoding="utf-8") as f:
        json.dump(reverse_graph, f)

    # Step 5: Generate mechanism-consistent textual corpus C_v via bounded-length random walks on Train Graph
    rng = random.Random(args.seed)
    corpus_path = os.path.join(pro_dir, "entity_corpus.jsonl")
    print(f"Sampling M={args.num_walks} random walks (max length T={args.max_walk_len}) per entity on Train Graph...")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for ent_idx in tqdm(range(len(entity2id)), desc="Building C_v"):
            text_corpus = sample_random_walks_for_entity(
                start_node=ent_idx,
                adj_list=adj_list,
                num_walks=args.num_walks,
                max_walk_len=args.max_walk_len,
                id2name=id2name_int,
                id2rel=id2rel,
                rng=rng,
            )
            record = {
                "entity_id": ent_idx,
                "raw_id": id2entity[ent_idx],
                "name": id2name_int[ent_idx],
                "text": text_corpus,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved mechanism-consistent corpus to {corpus_path}")


if __name__ == "__main__":
    main()
