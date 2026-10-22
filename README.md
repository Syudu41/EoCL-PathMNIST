# EoCL-PathMNIST
This is a replication of EoCL on the PathMNIST dataset. 

> [!IMPORTANT]
> Under Construction!!

### Dataset:

> [!NOTE] 
> Refer to [MedMNIST dataset](https://medmnist.com/).

```python 
pip install medmnist
```

### Workflow

- Image goes here


### Label-flip prototype (Stage 1 only)

```sh
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_prototype.py
```

The run downloads official 28×28 PathMNIST into `data/`, samples 40,000
training images, and creates stratified 30k/5k/5k splits. Seed 42 controls
pool sampling, splits, flips, and stochastic estimators. The four SVM/RF ×
m=1/5 configurations use the fixed settings in `status.md`; no tuning occurs.
All members vote on each test sample routed to their `y_given` bucket.

Outputs in `results/`:

- `results.csv`: four rows, with accuracy, precision, recall overall and for
  each `y_given` bucket; the full table is also printed.
- `flip_counts.csv`: exact floor(20%) flip counts per true class and split.
- `bucket_counts.csv`: poisoned/pristine counts per routing bucket and split,
  also printed as sanity lines, including test poisoned counts for interpreting zeros.
- `member_parameters.jsonl`: each member's hyperparameters and training count.
- `train.pkl`, `val.pkl`, `test.pkl`: pandas tables preserving uint8 image arrays
  and exactly `image`, `y_true`, `y_given`, `is_poisoned`. Validation is poisoned
  but excluded from model/scaler fitting, tuning, and metrics.

Only image pixels reach classifiers, scaled to [0,1]. SVM scalers fit on
individual training partitions. SHA-256 hashes are captured from clean image
bytes before poisoning. No mean-shift attack or federated training is included.

Run implementation checks with `.venv/bin/python -m unittest discover -s tests`.

### Visualize existing results

```sh
MPLCONFIGDIR=/tmp/eocl-matplotlib .venv/bin/python plot_results.py
```

This reads the saved CSVs without retraining. `results/overall_metrics.png`
compares the four runs; `results/per_class_metrics.png` shows the same three
metrics by `y_given` bucket, with test poisoned counts. Both figures also have
PDF versions for export. All panels use a common 0–100% scale.
