# Phase 2 Report: External Evaluation of DistilBERT Fake News Detection

**Project:** Fake News Detection using DistilBERT  
**Phase:** 2 — External dataset evaluation and error analysis  
**Dataset:** WELFake  
**Status:** Core evaluation completed; further validation recommended

## Executive summary

The saved DistilBERT classifier was evaluated on WELFake without retraining. After cleaning, 72,095 articles remained. An exact-text overlap audit found 44,899 WELFake rows matching the project's original Fake.csv or True.csv files (62.28%). The remaining 27,196 rows were evaluated separately. On that subset, accuracy was 65.35% and weighted F1 was 61.53%. Fake recall was 96.92%, while REAL recall was 33.84%. This indicates a substantial tendency to classify REAL articles as FAKE.

## 1. Objective and setup

The objective was to evaluate the saved Phase 1 DistilBERT model on an external dataset and investigate errors without retraining. Model labels are 0 = REAL and 1 = FAKE. Inference used CUDA, batch size 16, and max_length 256. Inputs combined title and body; missing titles were filled with empty strings and rows with missing text were removed. No training or fine-tuning was performed on WELFake.

## 2. Dataset preparation and labels

The WELFake CSV initially contained 72,134 rows, columns Unnamed: 0, title, text, label, 558 missing titles and 39 missing text values. After dropping rows with missing text and filling missing titles, 72,095 rows remained. The label convention was initially ambiguous. Exact matches to the project's source files showed that 23,482 WELFake rows matching Fake.csv had label 1, while 21,417 rows matching True.csv had label 0. This observed evidence supports WELFake 0 = REAL and 1 = FAKE for the evaluation. When publishing, cite and recheck the authoritative dataset documentation.

## 3. Exact overlap audit

Text was lowercased and whitespace-normalized; combined titles and article bodies were compared. WELFake rows checked: 72,095. Matches to Fake.csv: 23,482. Matches to True.csv: 21,417. Rows matching either source: 44,899 (62.28%). Rows matching both: 0. Rows without exact matches: 27,196 (37.72%). Fake.csv contains 23,481 rows, but 23,482 WELFake rows matched it; repeated WELFake rows can match the same source article, so these are row counts, not necessarily unique articles.

## 4. Main evaluation results

Full WELFake: 72,095 articles; accuracy 86.91%; weighted precision 89.07%; weighted recall 86.91%; weighted F1 86.67%. This includes exact-overlap rows and is not a clean independent evaluation.

## 5. Non-exact-overlap evaluation

Articles evaluated: 27,196. REAL support: 13,611. FAKE support: 13,585. Accuracy: 65.35%. Weighted precision: 75.55%. Weighted recall: 65.35%. Weighted F1: 61.53%. Macro F1: 61.54%.

## 6. Confusion matrix and class metrics

Confusion matrix (rows = actual, columns = predicted; order REAL, FAKE): [[4606, 9005], [418, 13167]]. REAL precision 91.68%, recall 33.84%, F1 49.43%, support 13,611. FAKE precision 59.39%, recall 96.92%, F1 73.65%, support 13,585. The model incorrectly classified 9,005 REAL articles as FAKE.

## 7. Comparison with Phase 1

Phase 1 validation: 7,727 articles, accuracy approximately 99.92%, weighted F1 approximately 99.92%. Full WELFake: 72,095 articles, accuracy 86.91%, weighted F1 86.67%. WELFake with exact matches removed: 27,196 articles, accuracy 65.35%, weighted F1 61.53%. This drop suggests the model's performance does not transfer equally to the non-exact-overlap subset. Possible explanations include source/topic differences, preprocessing differences, near-duplicates, or patterns learned from original data; this evaluation alone does not establish the cause.

## 8. Error and confidence analysis

There were 9,005 REAL-to-FAKE errors. Examples included articles about journalism practices, US political races, and mass-shooting reporting. These examples show confident misclassification of mainstream-news-style writing, but the examples themselves do not independently verify truthfulness; labels come from the dataset. Among REAL articles, correctly predicted REAL (n=4,606) had mean FAKE score 0.1525 and median 0.1036. REAL predicted FAKE (n=9,005) had mean FAKE score 0.9145 and median 0.9814. Softmax scores are model outputs, not calibrated probabilities that an article is false.

## 9. Article length analysis

REAL→REAL: 4,606 articles, median 1,027 words, mean 1,099.9. REAL→FAKE: 9,005, median 635, mean 786.3. FAKE→REAL: 418, median 385, mean 547.6. FAKE→FAKE: 13,167, median 404, mean 663.6. Correctly classified REAL articles were longer on average than REAL articles misclassified as FAKE. This is an association, not proof that length causes errors.

## 10. 256-vs-512 token diagnostic

A sample of 40 REAL articles was tested at max lengths 256 and 512. No predictions changed. Of 20 previously correct REAL predictions, 20 remained REAL; of 20 previously misclassified REAL articles, none changed to REAL. This small exploratory sample does not prove that longer inputs never help.

## 11. Limitations

1) Exact-match removal is not full deduplication; near-duplicates may remain. 2) Shared sources, publishers, and events may remain. 3) Label mapping should be documented with authoritative dataset provenance. 4) Dataset labels are not independent fact verification. 5) The token-length experiment used only 40 articles. 6) The model was not retrained; results describe the saved Phase 1 model.

## 12. Recommended next steps

1) Commit this report and the JSON metrics to version control. 2) Verify and cite authoritative WELFake label documentation. 3) Run near-duplicate analysis on the remaining 27,196 rows if compute permits. 4) Manually review a sample of labels and investigate source/topic artifacts. 5) Prefer a test set separated by source or time period where possible. 6) If retraining later, use group-aware splits and monitor balanced REAL/FAKE performance. 7) Describe the project as news classification/triage, not proof of truth or falsity.

## Conclusion

Phase 2's core external evaluation and error analysis are complete. The most informative result is 65.35% accuracy and 61.53% weighted F1 on 27,196 WELFake rows without exact normalized-text matches to the original source CSVs. The high overlap in the full dataset and low REAL recall are important findings. Because near-duplicates and shared-source overlap were not ruled out, the remaining subset should not be described as fully independent.

### Key metrics

| Evaluation | Articles | Accuracy | Weighted F1 |
|---|---:|---:|---:|
| Phase 1 validation | 7,727 | ~99.92% | ~99.92% |
| Full WELFake | 72,095 | 86.91% | 86.67% |
| WELFake without exact matches | 27,196 | 65.35% | 61.53% |

### Confusion matrix (rows = actual; columns = predicted; order REAL, FAKE)

| Actual / Predicted | REAL | FAKE |
|---|---:|---:|
| REAL | 4,606 | 9,005 |
| FAKE | 418 | 13,167 |

**Suggested reporting language:** We evaluated the saved DistilBERT model on WELFake. After removing 44,899 rows with exact normalized-text matches to the original source CSVs, the model achieved 65.35% accuracy and 61.53% weighted F1 on 27,196 remaining rows. Because near-duplicates and shared-source overlap were not ruled out, this subset should not be described as fully independent.