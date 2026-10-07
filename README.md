# CS313 Fraud Detection Benchmark

Demo phát hiện giao dịch thẻ tín dụng bất thường trên bộ dữ liệu ULB Credit Card Fraud Detection. Repository tập trung vào phần thực nghiệm và các output dùng để xây dựng slide cho ứng dụng fraud trong CS313 Project 1.

## Phương pháp

Notebook so sánh bốn mô hình trên cùng dữ liệu và preprocessing:

- DBSCAN
- HDBSCAN
- Isolation Forest
- Local Outlier Factor (LOF)

Nhãn `Class` không được truyền vào preprocessing, PCA, lựa chọn tham số hoặc model fitting. Nhãn chỉ được sử dụng sau khi mô hình chạy xong để đánh giá.

## Dataset

Bộ dữ liệu gốc gồm:

- 284.807 giao dịch
- 492 giao dịch fraud, khoảng 0,1727%
- `V1` đến `V28`: 28 thuộc tính PCA ẩn danh
- `Time` và `Amount`
- `Class`: `1 = Fraud`, `0 = Normal`

Sau khi xóa 1.081 dòng trùng, lần chạy gần nhất lấy ngẫu nhiên 60.000 dòng, trong đó có 98 fraud.

`creditcard.csv` không được commit. Dataset mặc định trên Kaggle:

```text
/kaggle/input/datasets/phtrnnam/cs313-namphu/creditcard.csv
```

## Pipeline

```text
creditcard.csv
  -> kiểm tra schema, missing value và duplicate
  -> xóa duplicate
  -> random sampling không dựa trên Class
  -> Time thành Time_sin và Time_cos
  -> Amount thành log1p(Amount)
  -> RobustScaler
  -> PCA 8D cho mô hình
  -> DBSCAN, HDBSCAN, Isolation Forest và LOF
  -> anomaly score hoặc noise label
  -> mở Class để đánh giá
  -> Precision, Recall, F1, AUPRC và runtime
```

PCA 2D được fit riêng và chỉ dùng để trực quan. Trong lần chạy hiện tại, PCA 8D giữ khoảng 66,47% phương sai; PCA 2D giữ khoảng 25,65%.

## Chạy trên Kaggle

1. Tạo Kaggle Notebook mới.
2. Chọn **File -> Import Notebook**.
3. Upload `fraud_benchmark/kaggle_fraud_benchmark.ipynb`.
4. Add Input chứa `creditcard.csv`.
5. Kiểm tra `DATA_PATH` trong cell cấu hình.
6. Chọn **Run All**.

Thông số mặc định:

```python
SAMPLE_SIZE = 60_000
PCA_COMPONENTS = 8
ALERT_RATE = 0.01
RANDOM_SEED = 42
DBSCAN_EPS = None
HDBSCAN_MIN_CLUSTER_SIZE = 50
LOF_NEIGHBORS = 35
IFOREST_TREES = 300
```

Nếu thiếu HDBSCAN, notebook sẽ thử cài package `hdbscan`. Nếu chạy quá lâu hoặc thiếu RAM, giảm `SAMPLE_SIZE` xuống 30.000.

Có thể chạy bản command-line trên Kaggle:

```bash
python fraud_benchmark/kaggle_fraud_benchmark.py \
  --data /kaggle/input/datasets/phtrnnam/cs313-namphu/creditcard.csv \
  --sample-size 60000 \
  --output /kaggle/working/fraud_density_outputs
```

## Kết quả hiện tại

Các model được so sánh tại alert budget 1%, tương ứng 600 cảnh báo trên 60.000 giao dịch.

| Model | Precision at 1% | Recall at 1% | F1 at 1% | AUPRC | Runtime |
|---|---:|---:|---:|---:|---:|
| DBSCAN | 2,83% | 17,35% | 4,87% | 2,24% | 52,83 giây |
| HDBSCAN | 1,83% | 11,22% | 3,15% | 1,20% | 47,04 giây |
| Isolation Forest | 1,33% | 8,16% | 2,29% | 1,88% | 2,88 giây |
| **LOF** | **3,67%** | **22,45%** | **6,30%** | **2,49%** | 10,88 giây |

Fraud prevalence trong sample là khoảng 0,1633%, nên AUPRC ngẫu nhiên cũng chỉ khoảng 0,1633%. LOF đạt AUPRC cao hơn baseline ngẫu nhiên khoảng 15,25 lần; DBSCAN cao hơn khoảng 13,74 lần.

### DBSCAN native output

Với `eps = 1,9699` và `min_samples = 16`:

- 4.200 giao dịch được đánh dấu là noise
- Bắt được 67 trong 98 fraud
- Recall: 68,37%
- Precision: 1,60%
- False positive: 4.133 giao dịch

DBSCAN minh họa tốt ý tưởng fraud có thể xuất hiện ở vùng mật độ thấp, nhưng native output tạo quá nhiều false positive để dùng trực tiếp như một hệ thống cảnh báo.

## Cấu trúc repository

```text
.
|-- fraud_benchmark/
|   |-- kaggle_fraud_benchmark.ipynb
|   |-- kaggle_fraud_benchmark.py
|   `-- outputs/
|       |-- data_summary.json
|       |-- experiment_config.json
|       |-- model_results.json
|       |-- model_comparison.csv
|       |-- k_distance.png
|       |-- dbscan_projection.png
|       |-- model_comparison.png
|       `-- confusion_matrices.png
|-- README.md
`-- .gitignore
```

`candidate_rankings.csv` và `creditcard.csv` được loại khỏi Git vì không cần thiết cho việc tiếp tục làm slide.

## Việc nên làm tiếp

1. Dùng DBSCAN làm mô hình chính để giải thích density-based và khái niệm noise.
2. Dùng LOF làm kết quả tốt nhất trong bảng benchmark hiện tại.
3. Trình bày so sánh tại cùng alert budget 1%.
4. Không sử dụng lại các số Precision 0,62, Recall 0,68 và AUPRC 0,44 từ demo cũ.
5. Cải thiện hình trước khi đưa vào slide:
   - Thu trục Y của `model_comparison.png` về khoảng 0 đến 0,25 và thêm nhãn số.
   - Zoom vùng elbow của `k_distance.png` quanh `eps` khoảng 1,97.
   - Giảm kích thước dấu noise trong `dbscan_projection.png`.
6. Nêu rõ đây là demo nghiên cứu trên sample ngẫu nhiên, chưa phải hệ thống fraud production hoặc real-time.

## Diễn giải kết quả

- Không kết luận DBSCAN tốt hơn mọi phương pháp.
- DBSCAN phù hợp để minh họa noise là output cần tìm.
- LOF cho ranking tốt nhất trong benchmark hiện tại.
- Isolation Forest chạy nhanh nhất nhưng chất lượng top 1% thấp hơn.
- HDBSCAN không cải thiện so với DBSCAN với cấu hình hiện tại.
