# KLaR — Tài liệu Kiến trúc & Quy trình Tái hiện (Agent Notes)

Tài liệu này lưu trữ các ý chính về phương pháp chuẩn bị dữ liệu và mã hóa cho hai nhánh của mô hình **KLaR**, dùng làm cơ sở để tái hiện và chạy trên các tập dữ liệu (dataset) mới.
- **Đặc tả kỹ thuật đầy đủ của mô hình KLaR** (toàn bộ công thức RGNN, Relation-Aware Additive Attention, Gated Fusion, InfoNCE Alignment Loss, DistMult Sparse MoE Decoder, Max-Margin Ranking Loss và Checklist chống Data Leakage) đã được lưu chi tiết tại file: [KLaR_Model_Spec.md](file:///d:/KLaR/KLaR_Model_Spec.md).

---

## 1. Chuẩn bị dữ liệu & Mã hóa cho 2 Nhánh (Dual-View Data Preparation)

### 1.1. Nhánh cấu trúc (Structural View)
- **Trích xuất đồ thị con bao quanh $k$-hop ($k$-hop Enclosing Subgraph)**:
  - Với mỗi thực thể mục tiêu $v$, trích xuất đồ thị con cục bộ $k$-hop: $G_v = (\mathcal{N}(v), \mathcal{E}_v)$.
  - $\mathcal{N}(v)$: Tập các nút láng giềng trong phạm vi $k$ bước nhảy từ $v$.
  - $\mathcal{E}_v$: Tập các cạnh quan hệ giữa các nút trong phạm vi $k$-hop.
- **Bảo toàn tính dị thể (Heterogeneity Preservation)**:
  - Giữ nguyên các loại thực thể (ví dụ: thuốc, gen, bệnh...) và nhãn loại quan hệ trên cạnh $r \in \mathcal{R}$.
  - Đưa trực tiếp $G_v$ vào mạng nơ-ron đồ thị quan hệ (**RGNN**) để truyền thông điệp theo từng loại quan hệ (relation-specific message passing), tạo ra biểu diễn cấu trúc cho thực thể $v$.

### 1.2. Nhánh văn bản (Textual View — Mechanism-Consistent Textualization)
Thay vì sử dụng mô tả tĩnh có sẵn của từng nút, KLaR biến đổi cấu trúc lân cận của thực thể $v$ thành ngữ cảnh văn bản nhất quán về mặt cơ chế sinh học qua 3 bước:

1. **Lấy mẫu đường đi ngẫu nhiên có độ dài giới hạn (Bounded-length Random Walks)**:
   - Bắt đầu từ thực thể $v$, lấy mẫu $M$ đường đi ngẫu nhiên trong cùng subgraph lân cận $G_v$: $\mathcal{P}_v = \{p_1, \dots, p_M\}$.
   - Siêu tham số (Hyperparameters):
     - $M$: Số lượng đường đi lấy mẫu trên mỗi thực thể.
     - $T$: Độ dài tối đa của mỗi đường đi ($\tau \le T$).

2. **Tuyến tính hóa theo mẫu quan hệ (Template-based Linearization)**:
   - Mỗi đường đi $p = (v_0 \xrightarrow{r_1} v_1 \dots \xrightarrow{r_\tau} v_\tau)$ được chuyển thành câu ngôn ngữ tự nhiên bằng cách ghép tên thực thể với mẫu từ/câu quy định sẵn $T_r$ cho từng loại quan hệ $r$:
     $$\text{seq}(p) = \text{Concat}(v_0, T_{r_1}, v_1, \dots, T_{r_\tau}, v_\tau)$$
   - *Ví dụ*:
     - `(MATR3) interacts (PABPC3) interacts (IGF2BP1) ... covaries (APLP1)`
     - `PLXNA1 regulates NOP56 downregulates Simvastatin upregulates MNT ...`

3. **Tạo kho ngữ cảnh, Token hóa & Mã hóa (Context Corpus, Tokenization & Encoding)**:
   - Tập hợp các chuỗi văn bản $\text{seq}(p)$ từ $\mathcal{P}_v$ tạo thành kho ngữ cảnh văn bản $C_v$ của thực thể $v$.
   - Token hóa $C_v$ thành chuỗi đầu vào $X_v$.
   - Đưa $X_v$ qua **bộ mã hóa câu đóng băng (Frozen Sentence Encoder)** kết hợp **mean pooling** để thu được vector biểu diễn văn bản $t_v$.
   - **Danh sách mô hình Encoder ưu tiên sử dụng khi tái hiện**:
     1. **PubMedBERT** (ví dụ: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`)
     2. **BERT** (ví dụ: `bert-base-uncased`)
     3. **Flan-T5** (ví dụ: `google/flan-t5-base` — sử dụng phần Encoder)
   - *(Lưu ý: Mô hình mặc định của bài báo là `Qwen3-Embedding-0.6B` tạm thời để sau, ưu tiên chạy và so sánh 3 mô hình trên trước).*

---

## 2. Thông tin & Quy tắc Giải mã Tên Thực thể cho các Tập dữ liệu Mới

### 2.1. Tập dữ liệu `kegg50k`
- **Tổng số node**: 16.201 thực thể sinh học, chia làm **5 loại thực thể** nhận biết qua tiền tố mã ID:

| Loại thực thể | Định dạng ID & Ví dụ | Giải mã & Tên thực thể | Nguồn & Link tải / API tra cứu |
| :--- | :--- | :--- | :--- |
| **Gen người (Gene)** | `HSA:<Số>`<br>VD: `HSA:351`, `HSA:51196` | Phần số là NCBI Entrez Gene ID (VD: `351` là gen *APP*, `51196` là *PLCE1*). | • Tra cứu: `https://www.ncbi.nlm.nih.gov/gene/351` hoặc `https://www.kegg.jp/entry/hsa:351`<br>• Toàn bộ tên gen người: `https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz`<br>• Danh sách gen từ KEGG REST: `https://rest.kegg.jp/list/hsa` |
| **Thuốc (Drug)** | `D#####`<br>VD: `D02042` | Mã trong KEGG DRUG (VD: `D02042` là *Olprinone hydrochloride hydrate*). | • Tra cứu: `https://www.kegg.jp/entry/D02042`<br>• Toàn bộ danh sách thuốc KEGG REST: `https://rest.kegg.jp/list/drug` |
| **Bệnh (Disease)** | `H#####`<br>VD: `H00056` | Mã trong KEGG DISEASE (VD: `H00056` là *Alzheimer disease*). | • Tra cứu: `https://www.kegg.jp/entry/H00056`<br>• Toàn bộ danh mục bệnh KEGG REST: `https://rest.kegg.jp/list/disease`<br>• Cây phân loại bệnh ICD-11: `https://rest.kegg.jp/get/br:br08403` |
| **Con đường sinh học (Pathway)** | `hsa#####`<br>VD: `hsa05010` | Mã trong KEGG PATHWAY (VD: `hsa05010` là *Alzheimer disease pathway*). | • Tra cứu: `https://www.kegg.jp/entry/hsa05010`<br>• Toàn bộ pathway người KEGG REST: `https://rest.kegg.jp/list/pathway/hsa` |
| **Mạng sinh học (Network)** | `N#####`<br>VD: `N00399` | Mã trong KEGG NETWORK (mô tả phân mạng con bị biến đổi trong bệnh). | • Tra cứu: `https://www.kegg.jp/entry/N00399`<br>• Danh sách network KEGG REST: `https://rest.kegg.jp/list/network` |

### 2.2. Tập dữ liệu `hetionet_aligned`
- **Tổng số node**: 19.930 thực thể chuẩn hóa, thuộc **3 loại thực thể chính**:

| Loại thực thể | Định dạng ID & Ví dụ | Giải mã & Tên thực thể | Nguồn & Link tải / API tra cứu |
| :--- | :--- | :--- | :--- |
| **Gen (Gene)** | `Gene::<Số>`<br>VD: `Gene::351`, `Gene::7316` | Phần số là NCBI Entrez Gene ID (VD: `351` là *APP*, `7316` là *UBC*). | • Tra cứu: `https://www.ncbi.nlm.nih.gov/gene/351`<br>• Tên, ký hiệu, mô tả: `https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz`<br>• Họ gen chuẩn HGNC: `https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt` |
| **Hợp chất / Thuốc (Compound)** | `Compound::DB#####`<br>VD: `Compound::DB00843` | Mã trong DrugBank (VD: `DB00843` là *Donepezil*). | • Tra cứu: `https://go.drugbank.com/drugs/DB00843`<br>• File nút gốc Hetionet (có đủ cả 3 loại node): `https://raw.githubusercontent.com/hetio/hetionet/master/describe/nodes/hetionet-v1.0-nodes.tsv` |
| **Bệnh (Disease)** | `Disease::DOID:#####`<br>VD: `Disease::DOID:10652` | Mã trong Human Disease Ontology (VD: `DOID:10652` là *Alzheimer's disease*). | • Tra cứu: `https://disease-ontology.org/?id=DOID:10652`<br>• Toàn bộ ontology định nghĩa & đồng nghĩa: `https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo` |

### 2.3. Tập dữ liệu `suppkg_v1` (và `ADInt`)
- **Tổng số node**: 38.915 thực thể (trích xuất tự động bằng công cụ SemRep từ văn bản PubMed về thực phẩm chức năng và hoạt chất):

| Loại ID | Định dạng & Ví dụ | Giải mã & Tên thực thể | Nguồn tra cứu / Tải file |
| :--- | :--- | :--- | :--- |
| **UMLS CUI** | `C#######`<br>VD: `C0001734`, `C0002395` | Mã khái niệm UMLS CUI (Concept Unique Identifier) (VD: `C0001734` là *aflatoxin*, `C0002395` là *Alzheimer's Disease*). | • Tra cứu: `https://www.ncbi.nlm.nih.gov/medgen/C0001734` hoặc `https://uts.nlm.nih.gov/uts/`<br>• Tên tiếng Anh chuẩn: file `cui2name.json` từ repo `https://github.com/zhang-informatics/ADInt`<br>• Tên đồng nghĩa y sinh: `https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/Thesaurus.OWL.zip` (nối qua thuộc tính CUI P207) |
| **iDISK Dietary Supplement** | `DC#######`<br>VD: `DC0002521` | Mã chế phẩm hoặc thành phần thực phẩm chức năng trong cơ sở dữ liệu iDISK. | • Tên nằm trong trường `terms` của đồ thị gốc SuppKG: repo `https://github.com/zhang-informatics/SemRep_DS`<br>• Khớp đồng nghĩa qua NCI Thesaurus (bỏ tiền tố `D` tra mã `C#######`) |
| **Gen số thuần (trong ADInt)** | Số thuần: `9882` | NCBI Entrez Gene ID (tương tự `kegg50k` và `hetionet_aligned`). | • Tra cứu: `https://www.ncbi.nlm.nih.gov/gene/9882` |
| **Phân loại ngữ nghĩa (SemType / SemGroup)** | Mã 4 ký tự: `phsu`, `dsyn`, `orch`... | Phân loại ngữ nghĩa chuẩn của UMLS (VD: `phsu` = Pharmacologic Substance, `dsyn` = Disease or Syndrome). | • Bảng 15 nhóm ngữ nghĩa: `https://lhncbc.nlm.nih.gov/semanticnetwork/download/SemGroups.txt`<br>• Định nghĩa từng SemType: `https://download.nlm.nih.gov/umls/kss/semnet/sn_current.tgz` |


