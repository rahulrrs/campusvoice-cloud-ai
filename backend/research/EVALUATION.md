# Dataset And Evaluation

## Dataset description

- Source files: `data/train.csv`, `data/test.csv`, `data/val.csv`
- Primary text field: `text`
- Primary target: `label_id` (complaint category class index)
- Auxiliary target: `priority_id_fixed` (priority class index)
- Data cleaning script: `scripts/clean_dataset.py`

The dataset combines campus complaint-style text and related public feedback samples. Labels are normalized into a shared taxonomy used by the routing model.

## Model comparison

Run:

```bash
python scripts/evaluate_baselines.py
```

This generates `outputs/baseline_eval.json` with:

- DistilBERT fine-tuned model (if local weights are available)
- Linear SVM baseline
- Logistic Regression baseline

## Fairness evaluation

Run:

```bash
python scripts/evaluate_fairness.py
```

This generates `outputs/fairness_eval.json` with subgroup label and priority metrics across:

- anonymity groups
- detected language groups
- complaint categories
- available user groups

The current evaluation is designed to surface disparities for review rather than certify fairness.

## Reported metrics

- `accuracy`
- `precision_macro`
- `recall_macro`
- `f1_macro`

These metrics support baseline comparison and error analysis for category classification.
