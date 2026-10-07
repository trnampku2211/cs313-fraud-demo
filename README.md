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

