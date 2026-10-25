# Phase 3: External Evaluation and Threshold Analysis

## 1. Objective

Investigate the external generalization of the DistilBERT fake-news
classifier using non-overlapping WELFake articles and examine the
effect of changing the FAKE prediction threshold.

## 2. Evaluation setup

- Model: Saved DistilBERT robust model
- Maximum input length: 256 tokens
- Evaluation articles: 27,196
- REAL articles: 13,611
- FAKE articles: 13,585
- Exact overlaps removed: 44,899
- Original threshold: 0.50
- Class mapping: 0 = REAL, 1 = FAKE

## 3. Threshold sensitivity results

| Threshold | Accuracy | REAL accuracy | FAKE accuracy | REAL false positives | FAKE false negatives |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 65.35% | 33.84% | 96.92% | 9005 | 418 |
| 0.70 | 68.04% | 40.70% | 95.43% | 8071 | 621 |
| 0.80 | 69.41% | 45.22% | 93.65% | 7456 | 863 |
| 0.90 | 70.39% | 51.16% | 89.66% | 6648 | 1405 |
| 0.95 | 71.19% | 56.88% | 85.52% | 5869 | 1967 |
| 0.99 | 72.75% | 74.20% | 71.29% | 3511 | 3900 |

## 4. Additional experiments

### Article length

REAL-article accuracy increased from 5.12% for articles with
1–128 tokens to 46.87% for articles exceeding 1,024 tokens.
FAKE-article accuracy remained between 96.13% and 98.96%
across the examined length groups.

These are associations and do not establish that length causes errors.

### Input format

On a balanced diagnostic sample of 200 articles:

| Input | Overall accuracy | REAL accuracy | FAKE accuracy |
|---|---:|---:|---:|
| Title + body | 66.5% | 36% | 97% |
| Body only | 53.5% | 13% | 94% |
| Title only | 55.5% | 13% | 98% |

The sample was used for diagnosis, not as an independent full-dataset estimate.

### Prediction confidence

At the default threshold, many incorrectly classified REAL articles
had high model confidence. Softmax confidence is not a guarantee
of factual correctness and may be poorly calibrated on external data.

## 5. Interpretation

The model's internal validation performance does not transfer reliably
to non-overlapping WELFake articles. Its external errors are asymmetric:
at threshold 0.50, many REAL articles are predicted as FAKE, while
relatively few FAKE articles are predicted as REAL.

Increasing the threshold reduces REAL false positives but increases
FAKE false negatives. The highest observed overall accuracy in this
threshold experiment occurred at 0.99, but this does not establish
0.99 as the best production threshold.

## 6. Limitations

- Thresholds were examined on the external evaluation set; selecting
  a final threshold from these results would introduce test-set tuning.
- The 200-article input-format experiment is small and diagnostic.
- Dataset labels were not independently fact-checked.
- The available dataset has no dedicated publisher/source column.
- Observed associations do not establish causation.

## 7. Recommended next steps

1. Preserve these results as a diagnostic baseline.
2. Investigate duplicate and dataset leakage risks.
3. Build source-aware and duplicate-aware train/validation/test splits
   where source metadata is available.
4. Select thresholds on validation data only.
5. Evaluate the chosen model and threshold on untouched test data.
6. Report per-class precision, recall, F1, confusion matrices, and
   calibration—not accuracy alone.
