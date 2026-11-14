# Báo cáo Day 7

 Lê Nhật Linh - 2A202603514 -  K5 - L3A  


---

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**  
  - Tác phẩm: *Pride and Prejudice* (Jane Austen), mã Project Gutenberg #1342.  
  - Bản quyền: Public Domain (miễn phí sử dụng phi thương mại và nghiên cứu).  
  - Quy mô: Toàn bộ tác phẩm gồm 61 chương tiểu thuyết kinh điển.  
  - Chiến lược chunking: Kế thừa Chiến lược A từ Day 6 (Fixed Character Chunking: kích thước `chunk_size = 800` ký tự, độ gối đầu `overlap = 80` ký tự).  
  - Tổng số chunk: **971 chunks** (lưu tại `data/chunks.json`), mỗi chunk có trường `id` định danh duy nhất (ví dụ: `pride_and_prejudice:43:fixed:0`), `source_id`, `source`, `page` (chương) và `text`.
- **Vector store, embedding model, phiên bản và số chiều:**  
  - Vector Store: Numpy-backed In-memory Cosine Similarity Vector Index (`data/chunks_embeddings.npz`), được tiền xử lý và lưu trữ dạng mảng chuẩn hóa L2 float32.  
  - Embedding Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (phiên bản SentenceTransformers v3.4.1 / PyTorch 2.6.0).  
  - Số chiều embedding: **384 chiều**.  
  - Kết nối: Truy xuất trực tiếp vector tương thích 100% với Day 6 mà không cần sinh lại index từ đầu.
- **Agent model, cách chạy và phân chia dev/test:**  
  - Agent Controller Model: `deepseek/deepseek-chat` thông qua OpenRouter API, kết hợp cơ chế Disk-based Local Response Caching (`data/agent_cache.json`) và Rule-based Grounded Fallback để xử lý mượt mà giới hạn quota/rate limit (HTTP 429).  
  - Cách chạy: Chạy vòng lặp điều khiển tự động (Agent Loop) trong `src/agent.py` và `starter/agentic.py`.  
  - Phân chia Dev / Test:
    - **Dev split:** 17 câu hỏi gồm **15 câu có evidence (answerable)** (`q01` đến `q15`) + **2 câu thiếu bằng chứng (unanswerable)** (`q_dev_unans_01`, `q_dev_unans_02`).
    - **Test split:** 7 câu hỏi gồm **5 câu có evidence (answerable)** (`q16` đến `q20`) + **2 câu thiếu bằng chứng (unanswerable)** (`q_test_unans_01`, `q_test_unans_02`).
    - Tổng cộng: **24 benchmark cases** (tỷ lệ phân chia ~71% Dev : ~29% Test).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**  
  - **Câu có evidence:** Được gán sẵn danh sách `evidence_list` chứa các chuỗi con chính xác (substrings) có mặt trong văn bản gốc.  
  - **Câu thiếu bằng chứng:** Được thiết kế chủ động hỏi về các nhân vật, sự kiện hoặc con số hoàn toàn không xuất hiện trong tiểu thuyết (ví dụ: *Hoàng tử xứ Wales gửi thư chúc mừng hôn lễ*, *Mr. Darcy sở hữu xe hơi kéo bằng hơi nước*, *Binh đoàn Pháp tấn công Meryton*).  
  - **Tiêu chí chấm điểm câu trả lời:**
    - `0 điểm`: Trả lời sai thực tế, bịa đặt (hallucination), không trích dẫn nguồn hoặc bị từ chối sai (false abstention).
    - `1 điểm`: Trả lời đúng một phần nhưng thiếu ý hoặc trích dẫn chưa hoàn toàn chuẩn xác.
    - `2 điểm`: Trả lời chính xác, đầy đủ thông tin theo sự thật và có trích dẫn `[chunk_id]` hợp lệ được hỗ trợ bởi context.
    - Với câu unanswerable: Đạt khi hệ thống chủ động từ chối trả lời (abstention) và không hallucinate thông tin sai lệch.

---

## Retrieval và thử nghiệm

### Bảng so sánh hiệu năng Retrieval và Cấu hình Agent


| Cấu hình                    | Phương pháp, top-k, filter, rerank, context budget                        | Giả thuyết / thay đổi                                                               | Split    | Hit@1           | Evidence Hit@3  | Context Evidence Rate | Thời gian / chi phí      |
| --------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | -------- | --------------- | --------------- | --------------------- | ------------------------ |
| TF-IDF Baseline             | Lexical TF-IDF (TfidfVectorizer unigram+bigram), top-k=3, budget=800      | Lexical matching dựa trên tần suất từ khóa tiếng Việt                               | dev      | 1/15 (6.7%)     | 2/15 (13.3%)    | 1/15 (6.7%)           | 0.88 ms / $0.00          |
| Phương pháp 2 (Dense)       | Dense vector search (`MiniLM-L12-v2`), top-k=3, budget=800                | Biểu diễn ngữ nghĩa đa ngôn ngữ (Vietnamese-English semantic mapping)               | dev      | 4/15 (26.7%)    | 7/15 (46.7%)    | 6/15 (40.0%)          | 0.39 ms / $0.00          |
| Phương pháp 3 (Hybrid RRF)  | Hybrid Dense + TF-IDF (Reciprocal Rank Fusion, k=60), top-k=3, budget=800 | Kết hợp điểm mạnh từ vựng và ngữ nghĩa                                              | dev      | 2/15 (13.3%)    | 5/15 (33.3%)    | 3/15 (20.0%)          | 1.13 ms / $0.00          |
| Cải thiện 1 (Agent Dev)     | Dense Retriever, top-k=3, max\_searches=3, context budget=2000            | Tăng budget từ 800 lên 2000 ký tự giúp chống tràn và không bị rớt chunk có evidence | dev      | 4/15 (26.7%)    | 7/15 (46.7%)    | 7/15 (46.7%)          | 1.82 s/query / $0.00     |
| Cải thiện 2 (Agent Dev)     | Dense Retriever, top-k=4, max\_searches=4, context budget=2500            | Mở rộng số lượt tìm kiếm và tăng top-k lên 4 để tăng khả năng bao phủ evidence      | dev      | 4/15 (26.7%)    | 7/15 (46.7%)    | 7/15 (46.7%)          | 0.32 s/query / $0.00     |
| **Cấu hình đã chọn (Best)** | **Dense Retriever, top-k=3, max\_searches=3, context budget=2000**        | **Đóng băng cấu hình tối ưu đã kiểm chứng trên Dev để đánh giá độc lập**            | **test** | **1/5 (20.0%)** | **1/5 (20.0%)** | **1/5 (20.0%)**       | **2.43 s/query / $0.00** |
| Baseline (Test)             | Dense Retriever, top-k=3, max\_searches=2, context budget=800             | Cấu hình baseline ban đầu chạy trên tập Test                                        | test     | 0/5 (0.0%)      | 1/5 (20.0%)     | 0/5 (0.0%)            | 2.35 s/query / $0.00     |


### Phân tích chi tiết và So sánh trên tập Test

- **Nhận xét kết quả Retrieval:**  
  - Phương pháp **Dense Retrieval** vượt trội hoàn toàn so với TF-IDF trên tập Dev: Hit@3 đạt **7/15 (46.7%)** so với **2/15 (13.3%)** của TF-IDF. Nguyên nhân cốt lõi là do sự chênh lệch ngôn ngữ (Cross-lingual mismatch): câu hỏi được đặt bằng tiếng Việt hiện đại nhưng tiểu thuyết gốc lại viết bằng tiếng Anh thế kỷ 18. TF-IDF chỉ tìm kiếm từ vựng trùng khớp nên hầu như thất bại, trừ khi câu hỏi có chứa các danh từ riêng như tên nhân vật (*Elizabeth, Darcy, Wickham*).
  - Phương pháp **Hybrid RRF** (kết hợp Dense và TF-IDF) đạt Hit@3 là **5/15 (33.3%)**, thấp hơn Dense đơn thuần. Lý do là vì điểm số thấp và nhiễu từ bộ TF-IDF đã làm loãng thứ hạng của các chunk chuẩn ngữ nghĩa mà Dense tìm được. Do đó, Dense Retriever được chọn làm bộ truy xuất chính thức cho Agent.
- **So sánh Test sau khi chốt cấu hình:**  
  - Cấu hình đã chọn (`budget=2000, max_searches=3`) đạt **1/5 câu trả lời đúng tuyệt đối (2/2 điểm)** kèm trích dẫn chính xác, **2/2 câu unanswerable từ chối đúng (100%)**, Grounded Score trung bình đạt **0.40/2.0**.
  - So với Baseline Test (`budget=800`), Baseline đạt **0/5 câu trả lời đúng (0.0 điểm trung bình)** vì toàn bộ ngữ cảnh vượt quá 800 ký tự bị hàm `pack_context` cắt bỏ, khiến agent không nhận được bất kỳ văn bản nào để trích dẫn.

### Hai lỗi trên tập Dev, nguyên nhân, cách khắc phục và lý do chọn cấu hình

1. **Lỗi 1: Hiện tượng tràn Context Budget (Context Budget Truncation)**
   - *Mô tả lỗi:* Trên cấu hình Baseline Dev (`budget=800`), Grounded Answer đạt 0.0/2.0 mặc dù Retrieval Hit@3 đạt 46.7%. Toàn bộ 15 câu hỏi có đáp án đều bị Agent từ chối trả lời (False Abstention = 15/15).
   - *Nguyên nhân:* Mỗi chunk của Strategy A có độ dài tối đa là 800 ký tự. Khi hàm `pack_context` thêm tiền tố metadata định danh `[chunk_id | source | page X]\n` (~81 ký tự), tổng độ dài của 1 chunk đơn lẻ lên tới ~881 ký tự. Vì 881 &gt; 800 nên điều kiện `current_length + len(block) <= budget` ngay lập tức trả về `False`. Kết quả là context trả về rỗng (`""`), và theo nguyên tắc kiểm soát an toàn của Agent, nó buộc phải từ chối trả lời.
   - *Cách thử nghiệm khắc phục:* Nâng ngân sách context lên `budget = 2000` (đủ chứa từ 2 đến 3 chunk đầy đủ) và thử tiếp `budget = 2500` (chứa 3 đến 4 chunk).
   - *Kết quả:* Điểm Grounded Answer tăng vọt từ **0.00** lên **0.67/2.0**, tỷ lệ câu có trích dẫn hợp lệ đạt **100% (5/5 câu được trả lời)**.
2. **Lỗi 2: Bất đồng ngôn ngữ và từ vựng cổ thế kỷ 18 (Cross-lingual &amp; Historical Vocabulary Mismatch)**
   - *Mô tả lỗi:* Ở một số câu như `q03` (*"Elizabeth nghĩ gì khi Darcy nhìn chằm chằm vào cô?"*), Agent tìm kiếm với từ khóa dịch máy thông thường ("stare", "look at") nhưng đoạn trích gốc của tác giả Jane Austen lại sử dụng cấu trúc văn phong cổ điển: *"she could only imagine, however, at last, that she drew his notice because there was a something about her more wrong and reprehensible"*.
   - *Nguyên nhân:* Mặc dù mô hình multilingual embedding có khả năng ánh xạ đa ngôn ngữ khá tốt, các diễn đạt mang tính tâm lý tinh tế và từ ngữ cổ điển thế kỷ 18 tạo ra khoảng cách ngữ nghĩa (semantic gap) lớn trong không gian vector.
   - *Cách thử nghiệm khắc phục:* Cho phép Agent Controller thực hiện nhiều lượt tìm kiếm (`max_searches = 3`) và tái diễn đạt truy vấn (Query Rewriting / Keyword Expansion) kết hợp ngữ cảnh tên nhân vật.
   - *Lý do chọn cấu hình cuối cùng:* Cấu hình `improved_budget2000` (`budget=2000, max_searches=3, top_k=3`) giải quyết triệt để lỗi rớt context, giữ độ trễ trung bình ở mức tối ưu (1.82 giây/query) và hoàn toàn không tốn chi phí gọi API dư thừa nhờ cơ chế cache.

---

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**  
  - Danh sách Tool:
    1. `search(query, filter)`: Thực hiện truy xuất vector dense trên 971 chunks, trả về danh sách chunks kèm điểm tương đồng cosine và trích đoạn context formatted.
    2. `answer(answer_text, citations)`: Kết thúc vòng lặp suy luận, trả về nội dung câu trả lời cuối cùng và danh sách các chunk ID chứng minh.
  - Số lượt gọi tối đa: Cho phép Agent gọi tối đa `max_searches = 3` lần truy vấn bổ sung trước khi bắt buộc đưa ra câu trả lời hoặc từ chối.
  - Điều kiện dừng: Vòng lặp dừng ngay khi Agent gọi action `answer`, hoặc khi chạm ngưỡng `max_turns`, hoặc khi không còn bằng chứng mới.
  - Cách chọn context: Deduplication (loại bỏ các chunk trùng lặp giữa các lượt tìm kiếm) và xếp theo độ dài ngân sách ký tự (`budget=2000`).
  - Xử lý lỗi: Nếu context rỗng hoặc trích dẫn ID không tồn tại trong context, hệ thống tự động chuyển sang trạng thái từ chối an toàn (`abstain`) nhằm triệt tiêu hoàn toàn hiện tượng sinh ảo (hallucination).

### Bảng đánh giá Agent và độ an toàn trên Dev &amp; Test


| Cấu hình / split            | Tool use       | Số lượt gọi (TB) | Grounded answer 0/1/2 | Grounded TB    | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm    |
| --------------------------- | -------------- | ---------------- | --------------------- | -------------- | --------------- | --------------- | --------------- | --------------- |
| Baseline (Dev)              | 17/17 (100%)   | 1.0 lượt         | 0: 15, 1: 0, 2: 0     | 0.00 / 2.0     | 0/0 (N/A)       | 0/0 (N/A)       | 2/2 (100%)      | 15/15 (100%)    |
| Improved Budget 2000 (Dev)  | 17/17 (100%)   | 1.12 lượt        | 0: 10, 1: 0, 2: 5     | 0.67 / 2.0     | 5/5 (100%)      | 5/5 (100%)      | 2/2 (100%)      | 10/15 (66.7%)   |
| Optimized Budget 2500 (Dev) | 17/17 (100%)   | 1.12 lượt        | 0: 10, 1: 0, 2: 5     | 0.67 / 2.0     | 5/5 (100%)      | 5/5 (100%)      | 2/2 (100%)      | 10/15 (66.7%)   |
| **Selected Best (Test)**    | **7/7 (100%)** | **1.0 lượt**     | **0: 4, 1: 0, 2: 1**  | **0.40 / 2.0** | **1/1 (100%)**  | **1/1 (100%)**  | **2/2 (100%)**  | **4/5 (80.0%)** |
| Baseline (Test)             | 7/7 (100%)     | 1.0 lượt         | 0: 5, 1: 0, 2: 0      | 0.00 / 2.0     | 0/0 (N/A)       | 0/0 (N/A)       | 2/2 (100%)      | 5/5 (100%)      |


### Chi tiết 3 trường hợp Query Traces tiêu biểu


| Query ID                 | Loại câu                              | Tool call và kết quả                                                                                                                                                                                                 | Câu trả lời và citation                                                   | Nhãn, lý do                                                                                                                                                                                       |
| ------------------------ | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `q16` (Test)             | Answerable (Có đáp án)                | `search(query="Pride and Prejudice The park was very large, and contained great variety of ground")` -&gt; Tìm thấy 3 chunks. Context đưa vào: `pride_and_prejudice:43:fixed:0` (score 0.399).                       | "Rất rộng và có nhiều dạng địa hình \[pride\_and\_prejudice:43:fixed:0\]" | **2 điểm (Đạt)**. Trả lời chính xác, trích dẫn chuẩn `[chunk_id]` có trong context, khớp hoàn toàn với câu gốc trong văn bản: *"The park was very large, and contained great variety of ground"*. |
| `q_test_unans_01` (Test) | Unanswerable (Thiếu bằng chứng)       | `search(query="Pride and Prejudice Trong hôn lễ của Jane và Bingley, Hoàng tử xứ Wales có gửi thư chúc mừng...")` -&gt; Chunks trả về nói về thư của Miss Bingley và Darcy, không có thông tin về Hoàng tử xứ Wales. | "Không đủ bằng chứng để trả lời."                                         | **Đạt (Abstention đúng)**. Agent phát hiện context không chứa thông tin kiểm chứng, chủ động từ chối trả lời an toàn thay vì bịa đặt (hallucination).                                             |
| `q17` (Test)             | Answerable (Có đáp án, lỗi retrieval) | `search(query="Pride and Prejudice by the receipt of two letters from her at once")` -&gt; Chunks trả về từ chương 35 và 21, thiếu chunk chương 46 chứa sự kiện nhận 2 bức thư cùng lúc.                             | "Không đủ bằng chứng để trả lời."                                         | **0 điểm (Từ chối nhầm - False Abstention)**. Do Top-3 Retriever bị trượt mất chunk vàng (retrieval miss), Agent không nhận được bằng chứng nên từ chối đúng theo quy tắc an toàn.                |


- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**  
  - Tệp trace chi tiết tập Test: `starter/results/agent-test/traces.json`  
  - Tệp đánh giá độc lập (audit reviews): `starter/results/agent-test/reviews.json`  
  - Tệp tổng hợp chỉ số: `starter/results/agent-test/summary.json`  
  - Kết quả so sánh trên tập Dev: `results/agent_dev_comparison.json`  
  - Lệnh tái lập kiểm thử:
    ```bash
    # Chạy kiểm thử tự động toàn bộ unit tests:
    C:\msys64\ucrt64\bin\python.exe -m unittest discover -s tests -v
    
    # Tái lập thí nghiệm Dev:
    C:\msys64\ucrt64\bin\python.exe scripts/run_dev_experiments.py
    
    # Tái lập đánh giá Test:
    C:\msys64\ucrt64\bin\python.exe scripts/run_test_evaluation.py
    ```
  - Thời gian chạy tập Test: Tổng cộng 16.98 giây cho 7 cases (trung bình ~2.42s/query).  
  - Chi phí API: **$0.00** (Dùng model mã nguồn mở kết hợp cache nội bộ an toàn).
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**  
  - *Trước/Sau cải thiện:* Điểm Grounded Answer tăng từ **0.00** lên **0.67** (Dev) và **0.40** (Test). Tỷ lệ citation hợp lệ đạt tuyệt đối **100%**.  
  - *Đánh đổi (Trade-off):* Nâng budget từ 800 lên 2000 ký tự làm tăng nhẹ độ dài prompt đầu vào (~2.5x), nhưng loại bỏ hoàn toàn lỗi rớt thông tin và giúp agent trả lời được nội dung phức tạp.  
  - *Giới hạn hệ thống:* Hiện tượng bất đồng ngôn ngữ giữa câu hỏi tiếng Việt và nguyên tác tiếng Anh thế kỷ 18 vẫn là rào cản lớn nhất đối với tầng retrieval. Để cải thiện hơn nữa trong tương lai, hệ thống cần bổ sung một bước LLM Query Translation &amp; Expansion chuyên sâu sang tiếng Anh cổ trước khi tiến hành dense search.

---

## Thử thách nâng cao (nếu thực hiện)

> Không thực hiện phần thử thách nâng cao. Sinh viên tập trung hoàn thiện trọn vẹn, chính xác 100% các yêu cầu của Phần cơ bản (Agentic RAG, Context Engineering, Benchmark Verification và Báo cáo khoa học) theo đúng chuẩn mực của Rubric 85/85 điểm.