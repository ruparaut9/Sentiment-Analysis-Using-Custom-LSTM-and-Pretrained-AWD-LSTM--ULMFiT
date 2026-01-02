# merge_metrics.py
# Collect metrics for all models and save as a Markdown table

models_metrics = {
    "ULMFiT": {
        "accuracy": 0.9115,
        "precision": 0.8934,
        "recall": 0.9352,
        "f1": 0.9138,
        "training_time": "approx 5 hours"
    },
    "Custom LSTM": {
        "accuracy": 0.8847,
        "precision": 0.8741,
        "recall": 0.8997,
        "f1": 0.8867,
        "training_time": "approx 2 hours"
    },
    "BERT Fine-Tuned": {
        "accuracy": 0.9236,
        "precision": 0.9084,
        "recall": 0.9428,
        "f1": 0.9253,
        "training_time": "approx 1.5 hours"
    }
}

report_path = r"D:\Projects\NLP_Sentiment_Analysis\models\evaluation_summary_table.md"

# Build Markdown table
header = "| Model | Accuracy | Precision | Recall | F1 Score | Training Time |\n"
separator = "|-------|----------|-----------|--------|----------|---------------|\n"

rows = ""
for model, metrics in models_metrics.items():
    rows += f"| {model} | {metrics['accuracy']} | {metrics['precision']} | {metrics['recall']} | {metrics['f1']} | {metrics['training_time']} |\n"

table = header + separator + rows

# Save to file
import os
os.makedirs(os.path.dirname(report_path), exist_ok=True)

with open(report_path, "w") as f:
    f.write("# Model Performance Comparison\n\n")
    f.write(table)

print("Merged metrics table saved to:", report_path)
