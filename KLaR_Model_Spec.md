# Đặc tả Kỹ thuật Tái hiện Mô hình KLaR (Bản Thuần Văn Bản)

Tài liệu này hệ thống hóa toàn bộ quy trình xử lý dữ liệu đầu vào, kiến trúc mô hình, hàm mục tiêu và giao thức đánh giá của bài báo **"KLaR: fusing knowledge graphs and language models for biomedical target discovery"** (*Bioinformatics*, 2026) dưới dạng đặc tả kỹ thuật chi tiết để chủ động triển khai bằng mã nguồn.

* **Kho mã nguồn gốc của tác giả:** `https://github.com/stardj/KLaR`

---

## 1. Phát biểu bài toán và nguyên tắc cốt lõi

* **Bài toán:** Dự đoán liên kết (link prediction) trên đồ thị tri thức y sinh $\mathcal{G} = (\mathcal{V}, \mathcal{R}, \mathcal{E})$, trong đó $\mathcal{V}$ là tập thực thể sinh học (thuốc, gen, bệnh...), $\mathcal{R}$ là tập loại quan hệ và $\mathcal{E} \subset \mathcal{V} \times \mathcal{R} \times \mathcal{V}$ là tập các bộ ba quan sát được. Mục tiêu là học hàm chấm điểm $f : \mathcal{V} \times \mathcal{R} \times \mathcal{V} \to \mathbb{R}$ sao cho các bộ ba đúng chưa quan sát được xếp hạng cao hơn các bộ ba sai.
* **Nguyên tắc nhất quán cơ chế (Mechanism-Consistent Dual-View):**
  * Tuyệt đối **không dùng văn bản hay kho ngữ liệu bên ngoài**.
  * Cả nhánh cấu trúc lẫn nhánh văn bản của một thực thể $v$ đều phải được trích xuất từ **cùng một đồ thị con cục bộ $k$-hop** xung quanh $v$.
  * Bộ mã hóa văn bản được giữ **đóng băng hoàn toàn** (chỉ đóng vai trò bộ trích xuất đặc trưng); toàn bộ năng lực suy luận quan hệ được chuyển sang khối hợp nhất (fusion) và bộ giải mã chuyên gia thưa (Sparse MoE).

---

## 2. Quy trình chuẩn bị và tiền xử lý dữ liệu

### 2.1. Chia tập dữ liệu và chống rò rỉ (Data Leakage)
* Dữ liệu đánh giá tiêu chuẩn gồm 3 đồ thị tri thức y sinh dị thể: **PharmKG**, **HetioNet** và **DTINet** (và mở rộng sang **kegg50k**, **hetionet_aligned**, **suppkg_v1**).
* Chia toàn bộ tập bộ ba theo tỷ lệ **80% Train / 10% Validation / 10% Test**.
* **Nguyên tắc bắt buộc:** Loại bỏ hoàn toàn mọi cạnh thuộc tập test (và không dùng cạnh tập test khi dựng đồ thị lân cận) trước khi trích xuất subgraph, lấy mẫu đường đi ngẫu nhiên và huấn luyện mô hình.

### 2.2. Chuẩn bị dữ liệu cho Nhánh cấu trúc (Structural View)
* Với mỗi thực thể truy vấn $v \in \mathcal{V}$, trích xuất đồ thị con bao quanh $k$-hop:
  $$G_v = (\mathcal{N}(v), \mathcal{E}_v)$$
  trong đó $\mathcal{N}(v)$ là tập các nút nằm trong phạm vi $k$ bước nhảy tính từ $v$, và $\mathcal{E}_v$ là tập các cạnh nối giữa chúng trên đồ thị huấn luyện.
* Bảo toàn đầy đủ nhãn loại quan hệ $r \in \mathcal{R}$ trên từng cạnh để phục vụ truyền thông điệp theo từng quan hệ trong mạng RGNN.

### 2.3. Chuẩn bị dữ liệu cho Nhánh văn bản (Textual View)
Biến đổi cấu trúc lân cận của $v$ thành ngữ cảnh văn bản qua 3 bước:
1. **Lấy mẫu đường đi ngẫu nhiên có giới hạn độ dài (Bounded-length Random Walks):** Bắt đầu từ nút $v$, lấy mẫu $M$ đường đi ngẫu nhiên có độ dài tối đa $T$ nằm trong chính subgraph cục bộ $G_v$, thu được tập đường đi $\mathcal{P}_v = \{p_1, \dots, p_M\}$.
2. **Tuyến tính hóa theo mẫu quan hệ (Template-based Linearization):** Với mỗi đường đi $p = (v_0 \xrightarrow{r_1} v_1 \dots \xrightarrow{r_\tau} v_\tau)$, ghép nối tên các thực thể xen kẽ với mẫu câu quy định sẵn $T_r$ ứng với từng loại quan hệ $r$:
   $$\text{seq}(p) = \text{Concat}(v_0, T_{r_1}, v_1, \dots, T_{r_\tau}, v_\tau)$$
   *Ví dụ định dạng chuỗi văn bản minh họa trong bài báo:* `(MATR3) interacts (PABPC3) interacts (IGF2BP1) ... covaries (APLP1)` hoặc `PLXNA1 regulates NOP56 downregulates Simvastatin upregulates MNT ...`.
3. **Tổng hợp kho ngữ cảnh và Token hóa:** Tập hợp toàn bộ chuỗi từ $M$ đường đi tạo thành ngữ cảnh văn bản $C_v$ cho thực thể $v$, sau đó token hóa thành chuỗi $X_v$ để đưa vào bộ mã hóa câu đóng băng.

### 2.4. Tạo mẫu âm khi huấn luyện (Negative Sampling)
* Với mỗi bộ ba dương $(h, r, t)$ trong tập huấn luyện, sinh các bộ ba âm $(h', r, t')$ bằng cách thay thế ngẫu nhiên **thực thể đầu ($h$)** hoặc **thực thể đuôi ($t$)** với xác suất bằng nhau ($50\% / 50\%$).
* Số lượng mặc định theo bài báo: **128 bộ ba âm cho mỗi bộ ba dương**.

---

## 3. Đặc tả kiến trúc mô hình KLaR

### 3.1. Nhánh I: Mã hóa văn bản đóng băng (Textual Encoding)
* **Bộ mã hóa:** Trong dự án tái hiện ưu tiên sử dụng **PubMedBERT**, **BERT**, và **Flan-T5** (sau đó mới đến `Qwen3-Embedding-0.6B`) ở trạng thái **đóng băng hoàn toàn** (không cập nhật trọng số, không truyền ngược gradient, không fine-tune, không sinh tự hồi quy).
* **Tổng hợp ngữ nghĩa (Mean Pooling):** Đưa chuỗi đã token hóa $X_v$ qua bộ mã hóa $\Phi : \mathcal{T} \to \mathbb{R}^{d_{\text{emb}}}$ và tính trung bình cộng các vector đầu ra để thu được vector biểu diễn văn bản $t_v \in \mathbb{R}^{d_t}$:
  $$t_v = \frac{1}{|X_v|} \sum_{x \in X_v} \Phi(x)$$
* *Lưu ý tối ưu khi tự lập trình:* Vì $C_v$ được trích xuất cố định từ đồ thị train và $\Phi$ đóng băng, nên chạy tiền trích xuất (pre-compute) toàn bộ ma trận $t_v$ cho mọi $v \in \mathcal{V}$ rồi lưu xuống đĩa trước khi huấn luyện để giảm mạnh bộ nhớ GPU và thời gian chạy.

### 3.2. Nhánh II: Mã hóa đồ thị cục bộ (Local Mechanism Graph Encoding – RGNN)
* Sử dụng mạng nơ-ron đồ thị quan hệ (RGNN) gồm $L$ lớp chạy trên đồ thị con $G_v$. Gọi $h_u^{(l)}$ là biểu diễn của nút $u$ tại lớp $l$.
* **Quy tắc cập nhật nút:**
  $$h_u^{(l+1)} = \sigma\left(W_0^{(l)} h_u^{(l)} + \sum_{r \in \mathcal{R}} \sum_{w \in \mathcal{N}_r(u)} \alpha_{u,r,w} W_r^{(l)} h_w^{(l)}\right)$$
  trong đó $\mathcal{N}_r(u)$ là tập các nút láng giềng nối với $u$ bằng quan hệ $r$, $W_0^{(l)}$ là ma trận tự biến đổi, và $W_r^{(l)}$ là ma trận biến đổi riêng theo loại quan hệ $r$.
* **Cơ chế Attention cộng theo quan hệ (Relation-Aware Additive Attention):**
  $$\alpha_{u,r,w} = \text{softmax}_{w \in \mathcal{N}_r(u)} \left(\text{LeakyReLU}\left(a_r^\top \left[W_r^{(l)} h_u^{(l)} \parallel W_r^{(l)} h_w^{(l)}\right]\right)\right)$$
  trong đó $a_r \in \mathbb{R}^{2d}$ là vector trọng số attention riêng cho quan hệ $r$ và $\parallel$ là phép ghép nối vector.
  * *Điểm then chốt:* Phép $\text{softmax}$ **chỉ chuẩn hóa trong nhóm láng giềng chia sẻ cùng một loại quan hệ $r$** nhằm giữ nguyên đặc thù ngữ nghĩa giữa các loại tương tác sinh học khác nhau.
* **Đầu ra nhánh cấu trúc:** Sau $L$ lớp, thu được vector cấu trúc $s_v = h_v^{(L)} \in \mathbb{R}^{d_s}$.

### 3.3. Khối hợp nhất và căn chỉnh cấu trúc – văn bản (Fusion & Alignment)
* **Chiếu về không gian chung:** Dùng hai mạng MLP riêng biệt $\phi_s$ và $\phi_t$ để chiếu $s_v$ và $t_v$ về cùng số chiều $d$.
* **Hợp nhất có cổng (Gated Integration):** Tính vector cổng động $\gamma_v \in (0, 1)^d$ để điều tiết mức độ đóng góp của cấu trúc và văn bản cho từng thực thể:
  $$\gamma_v = \sigma\left(W_\gamma [s_v; t_v] + b_\gamma\right)$$
  $$z_v = \gamma_v \odot \phi_s(s_v) + (1 - \gamma_v) \odot \phi_t(t_v)$$
  trong đó $b_\gamma \in \mathbb{R}^d$ là độ lệch học được, $[s_v; t_v]$ là phép ghép nối hai vector và $\odot$ là phép nhân từng phần tử.
* **Căn chỉnh tương phản (Contrastive Alignment):** Tối đa hóa thông tin tương hỗ giữa góc nhìn cấu trúc và góc nhìn văn bản bằng hàm mất mát InfoNCE trên từng mini-batch $\mathcal{B}$ với hệ số nhiệt độ $\tau$:
  $$\mathcal{L}_{\text{align}} = -\frac{1}{|\mathcal{B}|} \sum_{v \in \mathcal{B}} \log \frac{\exp\left(\text{sim}(\phi_s(s_v), \phi_t(t_v)) / \tau\right)}{\sum_{u \in \mathcal{B}} \exp\left(\text{sim}(\phi_s(s_v), \phi_t(t_u)) / \tau\right)}$$
  với $\text{sim}(\cdot, \cdot)$ là độ tương đồng cosine giữa hai vector sau khi chiếu.

### 3.4. Bộ giải mã Mixture-of-Experts thưa (Sparse MoE Decoder)
* Gồm tập $N$ chuyên gia $\{E_k\}_{k=1}^N$ (cấu hình đầy đủ trong bài báo là **$N = 64$ chuyên gia**). Theo sơ đồ kiến trúc ở Hình 1 của bài báo, đầu vào khối giải mã và đầu ra sau bước tổng hợp các chuyên gia được chuẩn hóa bằng **RMSNorm**.
* **Bước 1 – Tính điểm định tuyến (Router Logits):** Với bộ ba $(h, r, t)$, ghép nối vector hợp nhất của thực thể đầu $z_h$ với vector embedding học được của quan hệ $r_r \in \mathbb{R}^{d_r}$, rồi đưa qua mạng $\text{MLP}_g$ (gồm 2 lớp tuyến tính, chiều ẩn $d/2$, hàm kích hoạt ReLU):
  $$a(h, r) = \text{MLP}_g([z_h; r_r]) \in \mathbb{R}^N$$
* **Bước 2 – Chọn Top-$K$ chuyên gia và chuẩn hóa trọng số:** Chỉ giữ lại $K$ chuyên gia có giá trị logit $a_k(h, r)$ lớn nhất và chuẩn hóa lại xác suất bằng softmax trên đúng nhóm Top-$K$ đó:
  $$\pi_k(h, r) = \frac{\exp(a_k(h, r))}{\sum_{j \in \text{Top-k}(a(h, r))} \exp(a_j(h, r))}$$
* **Bước 3 – Chấm điểm chuyên gia và đầu ra cuối cùng:** Mỗi chuyên gia $E_k$ sở hữu bảng embedding quan hệ riêng $r_r^{(k)}$ và tính điểm tương tác song tuyến tính kiểu DistMult:
  $$E_k(z_h, z_t, r_r) = \langle z_h, r_r^{(k)} \odot z_t \rangle$$
  Điểm số đầu ra cuối cùng của bộ ba là tổ hợp tuyến tính có trọng số từ các chuyên gia được chọn:
  $$f(h, r, t) = \sum_{k \in \text{Top-k}(a(h, r))} \pi_k(h, r) E_k(z_h, z_t, r_r)$$

---

## 4. Hàm mục tiêu và quy trình huấn luyện

* **Tham số huấn luyện:** Tối ưu hóa end-to-end toàn bộ tham số của RGNN, khối hợp nhất (MLP chiếu & cổng), bảng embedding quan hệ, bộ định tuyến và các chuyên gia MoE (tổng quy mô khoảng **130M tham số huấn luyện được**, không tính bộ mã hóa văn bản đóng băng).
* **Hàm mục tiêu tổng hợp:**
  $$\mathcal{L} = \mathcal{L}_{\text{rank}} + \lambda_1 \mathcal{L}_{\text{align}} + \lambda_2 \mathcal{L}_{\text{moe}}$$
* **Thành phần 1 – Max-Margin Ranking Loss ($\mathcal{L}_{\text{rank}}$):** Đẩy điểm của bộ ba đúng cao hơn các bộ ba nhiễu ít nhất một khoảng biên $\gamma$:
  $$\mathcal{L}_{\text{rank}} = \sum_{(h, r, t) \in \mathcal{B}} \sum_{(h', r, t') \in \mathcal{N}} \max\left(0, \gamma + f(h', r, t') - f(h, r, t)\right)$$
* **Thành phần 2 – Alignment Loss ($\mathcal{L}_{\text{align}}$):** Hàm InfoNCE định nghĩa ở Mục 3.3.
* **Thành phần 3 – Điều chuẩn MoE ($\mathcal{L}_{\text{moe}}$):** Chống sập chuyên gia (expert collapse) và thúc đẩy phân bổ đều các chuyên gia:
  $$\mathcal{L}_{\text{moe}} = \lambda_{\text{load}} \mathcal{L}_{\text{load}} + \lambda_{\text{imp}} \mathcal{L}_{\text{imp}}$$
  trong đó $\mathcal{L}_{\text{load}}$ cân bằng tần suất thực tế mỗi chuyên gia được chọn trên mini-batch, còn $\mathcal{L}_{\text{imp}}$ phạt sự mất cân bằng tổng khối lượng xác suất định tuyến dồn vào một số ít chuyên gia.

---

## 5. Quy trình đánh giá đầu ra

* **Giao thức chuẩn (Standard Filtered Link Prediction):**
  * Với mỗi bộ ba $(h, r, t)$ trong tập validation/test, tính điểm cho tất cả các ứng viên trên **toàn bộ tập thực thể $\mathcal{V}$** (thay thế đuôi $(h, r, ?)$ và thay thế đầu $(?, r, t)$).
  * Trước khi tính thứ hạng của thực thể đúng, **lọc bỏ** toàn bộ các bộ ba đúng khác đã tồn tại trong đồ thị (train + val + test) khỏi danh sách ứng viên để tránh phạt sai mô hình.
* **Các chỉ số báo cáo:**
  * **MRR** (Mean Reciprocal Rank)
  * **Hits@1**, **Hits@3**, **Hits@10**
* **Độ ổn định thống kê:** Chạy huấn luyện và đánh giá trên **3 random seeds** khác nhau và báo cáo giá trị trung bình.

---

## 6. Bảng tổng hợp siêu tham số cần cấu hình

| Nhóm cấu hình | Tham số | Trạng thái trong bài báo |
| :--- | :--- | :--- |
| **Chia dữ liệu** | Tỷ lệ Train / Val / Test | Cố định: **80% / 10% / 10%** |
| **Negative Sampling** | Số mẫu âm mỗi bộ ba dương | Mặc định: **128** (50% head / 50% tail) |
| **Text Encoder** | Mô hình embedding câu đóng băng | Mặc định bài báo: **Qwen3-Embedding-0.6B**; Dự án ưu tiên: **PubMedBERT, BERT, Flan-T5** |
| **Subgraph & Walk** | Số bước nhảy $k$, số đường đi $M$, độ dài tối đa $T$, bảng mẫu câu $T_r$ | Tra trong Appendix 1.1 / GitHub hoặc tự tinh chỉnh (`max_path_len = 6`) |
| **Kiến trúc RGNN & Fusion** | Số lớp $L$, chiều vector $d_s, d_t, d, d_r$ | Tra trong Appendix 1.1 / GitHub (tổng ~130M tham số huấn luyện) |
| **MoE Decoder** | Tổng số chuyên gia $N$, số chuyên gia chọn Top-$K$ | Mặc định **$N = 64$** (bảng ablation thử 1, 2, 4, 8, 16, 32, 64); $K = 2$ hoặc $4$ |
| **Tối ưu hóa** | Biên $\gamma$, nhiệt độ $\tau$, trọng số $\lambda_1, \lambda_2, \lambda_{\text{load}}, \lambda_{\text{imp}}$, bộ tối ưu (Adam), learning rate, batch size | Tra trong Appendix 1.1 / GitHub hoặc tinh chỉnh trên tập Val |
| **Đánh giá** | Số lần chạy seed | **3 random seeds** (lấy trung bình) |

---

## 7. Checklist kiểm tra các lỗi thường gặp khi tự code

1. **Rò rỉ cạnh tập test:** Đảm bảo đồ thị dùng để trích xuất subgraph $G_v$, lấy mẫu random walk $\mathcal{P}_v$ và truyền thông điệp RGNN **chỉ chứa các cạnh của tập train**.
2. **Nhầm phạm vi Softmax trong RGNN:** Kiểm tra kỹ để $\alpha_{u,r,w}$ chỉ được chuẩn hóa softmax trên tập láng giềng $\mathcal{N}_r(u)$ có **cùng loại quan hệ $r$**, không gộp chung tất cả láng giềng mọi quan hệ vào cùng một phép softmax.
3. **Quên đóng băng bộ mã hóa văn bản:** Xác nhận toàn bộ tham số của Text Encoder không nhận gradient và không nằm trong optimizer.
4. **Sập chuyên gia (Expert Collapse) trong MoE:** Theo dõi phân phối tần suất chọn của 64 chuyên gia trong quá trình huấn luyện; nếu chỉ 1–2 chuyên gia được dùng liên tục, hãy kiểm tra lại công thức và hệ số $\lambda_{\text{load}}, \lambda_{\text{imp}}$.
5. **Xếp hạng khi hòa điểm và lọc bộ ba đúng:** Khi tính MRR/Hits@K trên toàn bộ tập $\mathcal{V}$, chỉ loại bỏ các bộ ba đúng *khác* (giữ nguyên thực thể mục tiêu cần dự đoán) và xử lý nhất quán trường hợp nhiều thực thể có cùng điểm số.
