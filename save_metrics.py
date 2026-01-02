metrics = {
    "accuracy": 0.9236,
    "precision": 0.9084,
    "recall": 0.9428,
    "f1": 0.9253,
    "training_time": "approx 1.5 hours"
}

# safer path string
report_path = r"D:\Projects\NLP_Sentiment_Analysis\models\evaluation_summary.txt"

import os
os.makedirs(os.path.dirname(report_path), exist_ok=True)

with open(report_path, "a") as f:
    f.write("BERT Fine-Tuned Metrics:\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")
    f.write("\n")  # add a blank line for readability
