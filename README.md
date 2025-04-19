# Report Detail

## 1. Introduction

The primary objective of this project was to build an automated email classification system for a support team, with integrated masking and demasking of personally identifiable information (PII). The overall workflow includes: (1) detecting and masking sensitive data in incoming emails, (2) classifying masked emails into predefined support categories (e.g., Billing Issues, Technical Support, Account Management), and (3) exposing the solution via a RESTful API. For classification, we selected a traditional machine learning pipeline combining TF–IDF vectorization with a Linear Support Vector Classifier (LinearSVC), and optimized its hyperparameters through grid search.

## 2. PII Masking Methodology

### 2.1 Requirements

Before any model processing, GDPR‑style compliance demands that all PII and PCI-related fields be obfuscated. We targeted eight entity types: `full_name`, `email`, `phone_number`, `dob`, `aadhar_num`, `credit_debit_no`, `cvv_no`, and `expiry_no`.

### 2.2 Implementation Details

- **NER-based detection**: We loaded spaCy’s `en_core_web_sm` model to detect `PERSON` entities. Each detected span was tagged as `full_name`.
- **Regex patterns**: For structured data (emails, phone numbers, dates, card numbers, etc.), we crafted regular expressions aligned to national formats:
  - Email: standard RFC‑style pattern
  - Phone: 10-digit Indian format (`\b\d{10}\b`)
  - DOB: `DD-MM-YYYY` or variants
  - Aadhar: four groups of four digits
  - Credit/Debit: 16-digit (allowing spaces/dashes)
  - CVV: 3 digits
  - Expiry: `MM/YY`
- **Masking logic**: We scanned the raw text, recorded spans & original values, then replaced each span (in reverse order to preserve indices) with the token `[entity_type]`. The utility function returned both the masked text and a list of entity metadata (start/end indices, classification, and original value).

### 2.3 Validation

Unit tests confirmed: overlapping patterns are handled safely, and the demasking function accurately restored original values in sequence. Manual spot-checks on a held‑out subset of 50 emails yielded 98% PII recall and 100% precision.

## 3. Classification Pipeline

### 3.1 Data Preprocessing

- Loaded `data/raw_emails.csv` into pandas DataFrame.
- For reproducibility, we seeded `random_state=42`.
- Split data into train (80%) and test (20%) sets using stratified sampling to maintain class balance.

### 3.2 Feature Extraction

We vectorized the masked email texts using `TfidfVectorizer` with:

- `strip_accents='unicode'`
- English `stop_words`
- Tuned `max_features` and `ngram_range` via grid search.

### 3.3 Model Selection

We considered two families:

1. **LinearSVC**: Effective for high-dimensional sparse text data. Fast to train and low latency at inference.
2. **Random Forest** (baseline): Robust but slower and less interpretable for text.

Based on preliminary experiments, LinearSVC outperformed Random Forest by \~5% in test set accuracy, so we focussed on tuning the SVM pipeline.

## 4. Hyperparameter Tuning

We wrapped the TF–IDF + LinearSVC pipeline in `GridSearchCV` with 5‑fold cross‑validation:

| Parameter             | Values tested  |
| --------------------- | -------------- |
| `tfidf__max_features` | [5000, 10000]  |
| `tfidf__ngram_range`  | [(1,1), (1,2)] |
| `clf__C`              | [0.1, 1, 10]   |

Key decisions:

- **Class weighting**: `class_weight='balanced'` to mitigate class imbalance.
- **Parallelism**: `n_jobs=-1` for full CPU utilization.

The grid search reported:

- **Best CV accuracy**: 0.81
- **Optimal params**: `{ tfidf__max_features: 10000, tfidf__ngram_range: (1,2), clf__C: 1 }`
- **Test accuracy**: 0.75 (20% relative improvement over baseline).

## 5. Deployment & API Integration

- **Model serialization**: We saved the optimal pipeline (`best_svm.pkl`) via `joblib.dump`.
- **FastAPI**: Implemented `POST /classify` endpoint that:
  1. Reads raw email text
  2. Runs `mask_pii` from `utils.py`
  3. Feeds masked text into the loaded SVM pipeline
  4. Returns JSON with original email, masked email, list of masked entities, and predicted category
- **Hosting**: The app runs on Uvicorn. For cloud deployment, we configured Hugging Face Spaces with a custom start command pointing to `uvicorn api:app`.

## 6. Challenges & Solutions

- **Ambiguous entity boundaries**: Overlapping regex and NER detections occasionally clashed. We resolved by sorting spans in descending order before replacement to avoid index shifts.
- **Class imbalance**: Some categories had <5% representation. Using `class_weight='balanced'` and stratified splitting reduced bias.
- **Compute constraints**: Grid search on large TF–IDF matrices was slow. We limited `max_features` to 10k and parallelized with `n_jobs`.

## 7. Conclusion

The final SVM-based solution achieves \~75% accuracy on unseen email data, meeting real‑world performance requirements. The modular codebase (masking utilities, training script, API layer) can be extended to deep‑learning models if higher accuracy is needed.

