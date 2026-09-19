# FarmOS V3.2 — Kaggle/Colab Training

Bản sạch để huấn luyện `TemporalIntentPolicyV32` trên Google Colab T4.

## Mục tiêu của bản này

- Giữ nguyên kiến trúc V3.2 đã chứng minh chơi được 720 bước.
- Không dùng pseudo-closed-loop replay metric để chọn checkpoint.
- AMP FP16 trên CUDA, theo dõi VRAM, throughput và ETA mỗi optimizer step.
- Giảm RAM bằng cách bỏ các JSON action/effect thô sau khi parse.
- Chạy một trainer trên một T4; không chạy seq32 và seq64 đồng thời.
- Dữ liệu và checkpoint khởi tạo nằm trong Kaggle Dataset private, không commit lên GitHub public.

## Colab: cài đặt

```python
!git clone https://github.com/CopyyQ/farmos.git
%cd farmos
!pip install -q -r requirements-colab.txt
!pip install -q -e . --no-deps
```

Upload `kaggle.json` trực tiếp vào `/content/farmos/kaggle/kaggle.json`, sau đó:

```python
!python scripts/prepare_data.py
```

`prepare_data.py` tự đặt `KAGGLE_CONFIG_DIR=/content/farmos/kaggle`. File credential được `.gitignore` chặn và không được commit lên GitHub.
## Train seq32

```python
!python scripts/colab_train.py \
  --config configs/t4_seq32.json \
  --run-name seq32_full1
```

Log `V3_BC_PROGRESS` hiển thị trực tiếp:

- `epoch_step / epoch_steps`
- `loss_total`
- `temporal_steps_per_sec`
- `eta_seconds`
- `gpu_memory_mb`
- `gpu_peak_memory_mb`

Kết quả nằm trong `runs/seq32_full1/` gồm `bc_best.pt`,
`bc_last.pt`, `history.jsonl` và `run_result.json`.

## Benchmark seq64

Chỉ chạy sau khi seq32 đã xong:

```python
!python scripts/colab_train.py \
  --config configs/t4_seq64_benchmark.json \
  --run-name seq64_full1
```
Seq32 dùng `batch_sequences=16`; seq64 dùng `batch_sequences=8`.
Hai cấu hình đều xử lý xấp xỉ 512 temporal rows mỗi optimizer update,
giúp A/B công bằng hơn và tránh tăng RAM gấp đôi.

## Validation

Profile mặc định là `fast`:

1. teacher-forced validation loss,
2. opening market continue/active gate,
3. checkpoint cuối epoch là checkpoint được giữ lại.

`free_history` cũ vẫn còn cho nghiên cứu nhưng không được coi là true
closed-loop và không dùng để chọn model ở profile Colab mặc định.

Sau khi train, chất lượng thực tế vẫn phải được quyết định bằng true
closed-loop 720-step benchmark, không phải chỉ bằng offline loss.

## Kiểm thử

```bash
PYTHONPATH=src:. pytest -q
```

Private dataset mặc định:
`ponschannel/farmos-v32-training-data`.
Có thể override bằng biến môi trường `FARMOS_KAGGLE_DATASET`.
