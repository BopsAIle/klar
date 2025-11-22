# KLaR: Fusing Knowledge Graphs and Language Models for Biomedical Target Discovery

<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#method">Method</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a> •
  <a href="#acknowledgments">Acknowledgments</a>
</p>

---

## Abstract

**KLaR** (Knowledge graphs and Language models for biomedical target discoveRy) is a novel framework that synergistically combines the structured knowledge from biomedical knowledge graphs with the contextual understanding capabilities of large language models (LLMs) for advanced target discovery in drug development. By bridging symbolic reasoning over knowledge graphs with the semantic comprehension of LLMs, KLaR enables more accurate identification and prioritization of therapeutic targets across various diseases.

**Key Contributions:**
- 🔗 **Hybrid Architecture**: Novel integration of knowledge graph embeddings with language model representations
- 🎯 **Target Discovery**: State-of-the-art performance in identifying novel therapeutic targets
- 🧬 **Biomedical Focus**: Specifically designed for biomedical applications with domain-aware components
- 📊 **Interpretable Results**: Provides explanations for target predictions based on both structured and unstructured evidence

---

## Key Features

- **Multi-modal Knowledge Integration**: Seamlessly fuses structured knowledge from biomedical KGs (e.g., DrugBank, STRING, DisGeNET) with textual information from scientific literature
- **Contextualized Embeddings**: Leverages pre-trained biomedical language models (BioBERT, PubMedBERT) enhanced with graph-structured information
- **Flexible Architecture**: Modular design allows for easy adaptation to different knowledge graphs and language models
- **Scalable Pipeline**: Efficient processing of large-scale biomedical data
- **Explainability**: Built-in mechanisms to interpret and validate predictions

---

## Installation

### Prerequisites
- Python >= 3.8
- numpy==1.24.1
- pytorch_lightning==2.1.2
- torch>=2.1.0
- tqdm==4.66.1
- transformers>=4.51.0
- accelerate>=0.20.0
- pandas>=2.1.0
- tensorboard>=2.10.0
- CUDA >= 11.3 (for GPU support)

### Setup

```bash
# Create a virtual environment using conda
conda create -n klar python=3.8
conda activate klar

# Install dependencies
pip install -r requirements.txt

# Prepare customized pre-trained checkpoints for each dataset
cd ./klar/texual_encoder_checkpoint
# DTI
python model_ckpt_creator.py -p './DTI' -n=8173 -mn 'Qwen/Qwen3-0.6B'
# Hetionet
python model_ckpt_creator.py -p './hetionet' -n=19930 -mn 'Qwen/Qwen3-0.6B'
# PharmKG
python model_ckpt_creator.py -p './PharmKG_8K' -n=7260 -mn 'Qwen/Qwen3-0.6B'

```

---

### Dataset
Download the datasets (DTINet, Hetionet, PharmKG). For each dataset, navigate to `./data/<dataset_name>` and create folders:
- raw_data/  
- pro_data/

Place the downloaded data in `raw_data/`, run the preprocessing script, and the processed files will be saved in `pro_data/`
```bash
python <dataset_name>_data_processing.py
```



## Quick Start

### Training
Create an `output/` folder in the project root to store checkpoint files after training.  
Then run the corresponding script to train the model on each dataset:

```bash
cd scripts
# DTINet
bash run_DTI.sh

# Hetionet
bash run_het.sh

# PharmKG-8K
bash run_pharm.sh

# Monitor training logs
tail -f -n 2000 log.log
```

The `PL_logs/` folder stores training log files during model operation. 


### Testing
Modify the `--test_model_path` parameter to point to your trained model, then run the corresponding script to perform testing:

```bash
cd scripts
# DTINet
bash test_DTI.sh

# Hetionet
bash test_het.sh

# PharmKG-8K
bash test_pharm.sh
```

---

## Method

KLaR integrates local topological structure with textualised biomedical context via a mechanism-specialised architecture. Formally, we address the problem of link prediction in a biomedical knowledge graph $G = (V, R, E)$, where $V$ is the set of biological entities, $R$ is the set of relation types, and $E \subseteq V \times R \times V$ is the set of observed triples. Our goal is to learn a scoring function $f : V \times R \times V \rightarrow \mathbb{R}$ that ranks unobserved true triples higher than false ones.

Distinct from recent trends relying on large-scale generative BioLLMs for inference, KLaR decouples semantic encoding from reasoning. It utilises a frozen pretrained encoder solely for token-level representation, shifting the computational burden of reasoning to a compact, mathematically grounded fusion encoder and a Mixture-of-Experts (MoE) decoder.

### Model Architecture Overview

The inference workflow processes a query entity $v$ through two parallel views: (1) **Structural View**, where a relational GNN encodes the $k$-hop local mechanism graph $G_v$; and (2) **Textual View**, where mechanism-consistent paths are linearised and embedded. These views are projected into a shared latent space via a gated fusion mechanism. Finally, to capture the high heterogeneity of biological interactions (e.g., distinct patterns in metabolic pathways vs. gene regulation), we employ a sparse MoE decoder to predict link probabilities.

### View I: Mechanism-Consistent Textual Encoding

To capture the semantic context without the latency of generative decoding, we employ a frozen pretrained textual encoder $\phi : \mathcal{T} \rightarrow \mathbb{R}^{d_{emb}}$, where $\mathcal{T}$ is the token space, $d_{emb}$ is the dimension of textual embedding. In this work, we instantiate $\phi$ with Qwen3-Embedding, though the framework is agnostic to the specific encoder.

**Neighbourhood Textualisation.** Instead of utilizing external unstructured text which may introduce hallucinations, we generate a textual corpus $\mathcal{C}_v$ strictly derived from the graph topology. We sample a set of bounded-length random walks starting from $v$: $\mathcal{P}_v = \{p_1, \ldots, p_M\}$. Each path $p = (v_0 \xrightarrow{r_1} v_1 \cdots \xrightarrow{r_T} v_T)$ is linearised using relation-specific templates $T_r$: $\text{seq}(p) = \text{Concat}(v_0, T_{r_1}, v_1, \ldots, T_{r_T}, v_T)$. This ensures the textual input is mechanism-consistent with the underlying biological logic.

**Semantic Aggregation.** Let $X_v$ be the tokenised sequence of $\mathcal{C}_v$. The $d_t$ is the dimension of the static semantic representation $\mathbf{t}_v$, and $\mathbf{t}_v$ is obtained via mean-pooling over the frozen encoder outputs:

$$\mathbf{t}_v = \frac{1}{|X_v|} \sum_{x \in X_v} \phi(x) \in \mathbb{R}^{d_t}$$

Crucially, gradients are not backpropagated through $\phi$, ensuring training efficiency.

### View II: Local Mechanism Graph Encoding

To capture the topological signals, we extract the $k$-hop enclosing subgraph $G_v = (\mathcal{N}(v), E_v)$ for target entity $v$. We employ a Relational Graph Neural Network (RGNN) to encode this structure.

Let $\mathbf{h}_u^{(\ell)}$ denote the embedding of node $u$ at layer $\ell$. The update rule aggregates messages from relation-specific neighbourhoods:

$$\mathbf{h}_u^{(\ell+1)} = \sigma\left(W_0 \mathbf{h}_u^{(\ell)} + \sum_{r \in R} \sum_{w \in \mathcal{N}_r(u)} \alpha_{u,r,w} W_r^{(\ell)} \mathbf{h}_w^{(\ell)}\right)$$

where $\mathcal{N}_r(u)$ denotes neighbours connected via relation $r$, $W_r^{(\ell)}$ represents relation-specific transformation matrices, and $\alpha_{u,r,w}$ is the attention coefficient capturing the importance of the specific interaction $(u, r, w)$ in the biological pathway. The final structural embedding is $\mathbf{s}_v = \mathbf{h}_v^{(L)} \in \mathbb{R}^{d_s}$.

### Structure-Text Fusion and Alignment

Biomedical entities often exhibit a "modality gap" between their graph topology and semantic description. We bridge this using a learnable gated fusion.

**Gated Integration.** We project both representations into a unified dimension $d$ via MLPs $\psi_s$ and $\psi_t$. A gate $\gamma_v \in (0, 1)$ dynamically balances the information flow:

$$\gamma_v = \sigma(W_g [\mathbf{s}_v; \mathbf{t}_v] + b_g)$$

$$\mathbf{z}_v = \gamma_v \psi_s(\mathbf{s}_v) + (1 - \gamma_v) \psi_t(\mathbf{t}_v)$$

**Contrastive Alignment.** To enforce semantic consistency between the two views, we maximise the mutual information between $\mathbf{s}_v$ and $\mathbf{t}_v$ using an InfoNCE-style objective:

$$\mathcal{L}_{align} = -\mathbb{E}_{v \in V} \left[ \log \frac{\exp(\text{sim}(\psi_s(\mathbf{s}_v), \psi_t(\mathbf{t}_v))/\tau)}{\sum_{u \in B, u \neq v} \exp(\text{sim}(\psi_s(\mathbf{s}_v), \psi_t(\mathbf{t}_u))/\tau)} \right]$$

where $B$ is the mini-batch and $\tau$ is a temperature parameter.

### Mechanism-Specialised Mixture-of-Experts Decoder

Biomedical relations are highly heterogeneous (e.g., *treats* is many-to-many, while *causes* is causal and directional). A single scoring function often struggles to fit this multi-modal distribution. We address this via a Mixture-of-Experts (MoE) decoder.

We define a set of $N$ experts $\{E_k\}_{k=1}^N$, where each expert is a scoring function (e.g., a simplified DistMult or TransE block). For a given triple $(h, r, t)$, a sparse router network $\mathcal{G}(\cdot)$ selects the Top-$K$ most relevant experts based on the relation type and entity context:

$$f(h, r, t) = \sum_{k \in \text{Top-}K(\mathcal{G}(h, r))} \mathcal{G}(h, r)_k \cdot E_k(\mathbf{z}_h, \mathbf{z}_t, \mathbf{r}_r)$$

Here, $\mathcal{G}(h, r) = \text{Softmax}(W_g [\mathbf{z}_h; \mathbf{r}_r])$. This allows KLaR to dynamically allocate capacity: complex molecular interactions may activate specific experts, while clinical associations activate others.

To prevent mode collapse (where one expert dominates), we add a load-balancing loss $\mathcal{L}_{moe}$.

### Optimisation Objective

The model is trained end-to-end by minimising a joint objective comprising the max-margin ranking loss $\mathcal{L}_{rank}$, the alignment loss, and the expert regularisation:

$$\mathcal{L} = \mathcal{L}_{rank} + \lambda_1 \mathcal{L}_{align} + \lambda_2 \mathcal{L}_{moe}$$

We optimise all parameters of the encoder and decoder, while the Qwen3-Embedding module remains frozen throughout.

---

## Experiments

### Experimental Setup

**Datasets.** We evaluate KLaR on three heterogeneous biomedical knowledge graphs: PharmKG, HetioNet, and DTINet. All datasets are split into 80%/10%/10% for training/validation/testing under the filtered link prediction protocol.

**Negative Sampling.** For each positive triple $(h, r, t)$, we generate corrupted negatives by replacing either the head or the tail with equal probability. Unless otherwise stated, we sample 128 negative triples per positive example.

**Evaluation Metrics.** We report Mean Reciprocal Rank (MRR), Hits@1, Hits@3, and Hits@10, following the filtered setting. All results are averaged over three random seeds.

**Text embedding instantiation.** For all KLaR experiments, we use Qwen3-Embedding-0.6B as a frozen sentence-embedding model. Only its embedding output are used; no transformer layers or contextual inference are involved. All contextual reasoning is performed solely by our lightweight fusion encoder.

### Baselines

We compare KLaR with four categories of representative methods:

- **Structural KGE models**: TransE, DistMult, ComplEx, RESCAL, RotatE, Analogy, TransR, ProjE
- **Neural KGE / GNN-based models**: ConvE, ConvKB, R-GCN, HittER, STaR
- **Large-scale language models**: BioWordVec and domain-specific LLMs (Qwen3-8B, LLaMA2-7B, DeepSeek)
- **KG–LLM hybrid and multimodal biomedical models**: FuseLinker, KG-LLM, BioBLP, DTD-GNN

The first two groups focus purely on graph structure; text-enhanced models quantify the benefits of contextual biomedical language modeling; and KG–LLM hybrids represent the strongest non-KLaR baselines due to their explicit integration of structural and textual information.

---

## Results

### Overall Performance

KLaR achieves the best performance on all datasets and all metrics. The following table reports the benchmark results on PharmKG, HetioNet, and DTINet:

| Model | PharmKG | | | | HetioNet | | | | DTINet | | | |
|-------|---------|-|-|-|----------|-|-|-|--------|-|-|-|
| | MRR | H@1 | H@3 | H@10 | MRR | H@1 | H@3 | H@10 | MRR | H@1 | H@3 | H@10 |
| **Structural KGE Models** |
| TransE | 0.091 | 0.034 | 0.092 | 0.198 | 0.027 | 0.002 | 0.022 | 0.070 | 0.102 | 0.035 | 0.095 | 0.218 |
| DistMult | 0.071 | 0.023 | 0.062 | 0.157 | 0.029 | 0.006 | 0.022 | 0.067 | 0.013 | 0.003 | 0.008 | 0.025 |
| ComplEx | 0.107 | 0.046 | 0.110 | 0.225 | 0.070 | 0.029 | 0.069 | 0.148 | 0.138 | 0.063 | 0.149 | 0.326 |
| RotatE | 0.071 | 0.023 | 0.057 | 0.161 | 0.023 | 0.002 | 0.012 | 0.054 | 0.010 | 0.001 | 0.004 | 0.014 |
| **Neural KGE / GNN Models** |
| ConvE | 0.086 | 0.038 | 0.087 | 0.169 | 0.075 | 0.032 | 0.071 | 0.155 | 0.127 | 0.047 | 0.141 | 0.241 |
| R-GCN | 0.067 | 0.027 | 0.062 | 0.139 | 0.030 | 0.011 | 0.021 | 0.052 | 0.074 | 0.029 | 0.068 | 0.142 |
| **Text-Enhanced Models** |
| BioWordVec | 0.121 | 0.057 | 0.112 | 0.250 | 0.080 | 0.019 | 0.056 | 0.220 | 0.188 | 0.076 | 0.186 | 0.475 |
| Qwen3-8B | 0.159 | 0.069 | 0.170 | 0.383 | 0.125 | 0.054 | 0.123 | 0.277 | 0.214 | 0.080 | 0.219 | 0.546 |
| LLaMA2-7B | 0.158 | 0.068 | 0.169 | 0.350 | 0.124 | 0.053 | 0.122 | 0.274 | 0.213 | 0.079 | 0.218 | 0.543 |
| DeepSeek | 0.162 | 0.068 | 0.180 | 0.360 | 0.127 | 0.055 | 0.125 | 0.292 | 0.218 | 0.081 | 0.222 | 0.607 |
| **KG–LLM Hybrid Models** |
| FuseLinker | 0.169 | 0.075 | 0.177 | 0.372 | 0.133 | 0.057 | 0.137 | 0.286 | 0.232 | 0.090 | 0.246 | 0.605 |
| KG-LLM | 0.166 | 0.072 | 0.174 | 0.379 | 0.131 | 0.056 | 0.135 | 0.289 | 0.233 | 0.088 | 0.243 | 0.600 |
| BioBLP | 0.145 | 0.061 | 0.150 | 0.330 | 0.121 | 0.051 | 0.120 | 0.268 | 0.210 | 0.077 | 0.215 | 0.538 |
| DTD-GNN | - | - | - | - | - | - | - | - | 0.236 | 0.087 | 0.241 | 0.598 |
| **KLaR (Ours)** | **0.182** | **0.089** | **0.194** | **0.396** | **0.146** | **0.069** | **0.151** | **0.306** | **0.254** | **0.103** | **0.261** | **0.621** |

**Key Observations:**
- On PharmKG, KLaR improves MRR from 0.169 (FuseLinker) and 0.162 (DeepSeek) to 0.182, and raises Hits@10 to 0.396.
- On HetioNet, KLaR further improves the strongest baseline (FuseLinker, MRR 0.133) to 0.146.
- On DTINet, KLaR attains an MRR of 0.254, outperforming FuseLinker (0.232) and DTD-GNN (0.236), and achieves a Hits@10 of 0.621.
- KLaR surpasses all LLM baselines despite using only the 0.6B embedding variant (Qwen3-Embedding-0.6B) and a lightweight fusion model, showing that explicit local mechanism encoding and structure–text alignment can outperform full-scale LLM reasoning for biomedical KG completion.

### Ablation Study

To better understand the contribution of each component in KLaR, we conduct extensive ablation studies on all three datasets:

| Ablation | PharmKG | | | HetioNet | | | DTINet | | | |
|----------|---------|-|-|----------|-|-|--------|-|-|-|
| | MRR | H@1 | H@3 | H@10 | MRR | H@1 | H@3 | H@10 | MRR | H@1 | H@3 | H@10 |
| **Expert Count Variation** |
| 1 Expert | 0.155 | 0.062 | 0.157 | 0.364 | 0.113 | 0.045 | 0.108 | 0.254 | 0.203 | 0.071 | 0.199 | 0.526 |
| 4 Experts | 0.163 | 0.069 | 0.170 | 0.370 | 0.119 | 0.049 | 0.118 | 0.259 | 0.224 | 0.082 | 0.229 | 0.580 |
| 16 Experts | 0.171 | 0.075 | 0.181 | 0.385 | 0.126 | 0.053 | 0.124 | 0.280 | 0.227 | 0.085 | 0.233 | 0.585 |
| 32 Experts | 0.170 | 0.075 | 0.180 | 0.385 | 0.128 | 0.056 | 0.127 | 0.278 | 0.239 | 0.094 | 0.251 | 0.580 |
| **Component Ablation** |
| w/o MoE | 0.168 | 0.073 | 0.178 | 0.378 | 0.121 | 0.052 | 0.124 | 0.275 | 0.230 | 0.086 | 0.237 | 0.588 |
| w/o KGE | 0.162 | 0.067 | 0.164 | 0.374 | 0.118 | 0.050 | 0.118 | 0.263 | 0.217 | 0.078 | 0.220 | 0.564 |
| w/o TextE | 0.169 | 0.075 | 0.181 | 0.382 | 0.132 | 0.056 | 0.135 | 0.290 | 0.236 | 0.090 | 0.246 | 0.603 |
| w/o L_align | 0.170 | 0.076 | 0.182 | 0.384 | 0.133 | 0.057 | 0.137 | 0.291 | 0.238 | 0.091 | 0.247 | 0.606 |
| **KLaR (64 Experts)** | **0.182** | **0.089** | **0.194** | **0.396** | **0.146** | **0.069** | **0.151** | **0.306** | **0.254** | **0.103** | **0.261** | **0.621** |

**Key Findings:**
- **Effect of the MoE**: Removing the MoE decoder leads to a notable drop in performance (MRR decreases from 0.182 to 0.168 on PharmKG and from 0.254 to 0.230 on DTINet), confirming that sparse expert specialization is beneficial for modeling diverse biomedical relation patterns.
- **Effect of structural and textual branches**: Removing the structural embedding module yields a clear MRR drop (0.182 → 0.162 on PharmKG), underlining that structural signals remain the backbone of KG reasoning. Removing the text embedding branch also reduces performance (0.182 → 0.169 on PharmKG), showing that textual information provides complementary evidence.
- **Effect of structural–textual fusion**: Disabling the alignment loss consistently weakens results, indicating that explicit structural–textual alignment stabilizes the fusion module and prevents over-reliance on any single modality.
- **Effect of expert count**: Varying the number of experts from 1 to 64 shows steady improvement in MRR and Hits@K, especially when moving from 1 to 16 experts, after which gains gradually saturate.

### Biological Case Studies

To evaluate whether KLaR can recover biologically meaningful relations absent at training time, we conduct two qualitative case studies. In both modules, the dashed edges are missing from the original KGs yet ranked highly by KLaR. Notably, these predicted links correspond to mechanistic connections only recently characterized in the biomedical literature, providing a retrospective form of temporal validation.

**Case I: Mitochondrial axis linking PCOS and Parkinson's disease.** Channel A centers on PCOS, TSFM, and Parkinson's disease. Both PCOS and PD are increasingly recognized as disorders with pronounced mitochondrial abnormalities, including oxidative stress, respiratory-chain defects, and impaired mitophagy. TSFM encodes a mitochondrial translation factor essential for oxidative phosphorylation. Although the training KGs contain no direct PCOS–PD or TSFM-related mitochondrial edges, KLaR predicts PCOS–TSFM and TSFM–PD associations and reconstructs a coherent mitochondrial module. This aligns with emerging evidence suggesting shared mitochondrial and metabolic disruptions across the two diseases.

**Case II: Metabolic nuclear receptor axis linking NAFLD and type 2 diabetes.** Channel B examines NAFLD, T2DM, and regulators PPARG and ADIPOQ. PPAR governs adipocyte differentiation and insulin sensitivity, while adiponectin protects against hepatic steatosis and metabolic dysfunction. PPAR agonists such as pioglitazone improve NASH/NAFLD in clinical trials. Although several NAFLD–PPARG/ADIPOQ edges are absent or sparse in the KG, KLaR predicts NAFLD–PPARG, NAFLD–ADIPOQ and PPARG–T2DM links, forming a coherent metabolic axis consistent with established mechanisms. This suggests that KLaR can surface clinically relevant pathways involving druggable nuclear receptors.


---

## Project Structure

```
KLaR/
├── data/                 # Datasets and knowledge graphs
├── klar/                 # Main source code
│   ├── model/            # Model architectures
│   ├── data/             # Data loaders and processors
│   ├── graph_encoder_checkpoint/   # graph encoder checkpoint
│   ├── qwen3_checkpoint/           # textual encoder checkpoint
│   └── __init__.py
├── scripts/             # Training and evaluation scripts
├── main.py              # Entry point for training and evaluation
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.12+-red.svg" alt="PyTorch">
</p>
