import pandas as pd
import json
import os

def validate_datasets():
    datasets = {
        "raw": "C:/Users/ASUS/Documents/AITF-2026/PKL/Datasets/raw_review.csv",
        "sentiment": "C:/Users/ASUS/Documents/AITF-2026/PKL/Datasets/sentiment_preprocessed.csv",
        "topic": "C:/Users/ASUS/Documents/AITF-2026/PKL/Datasets/topic_preprocessed.csv"
    }
    
    report = {"datasets": {}}
    
    for name, path in datasets.items():
        try:
            # Check encoding consistency by attempting to read with utf-8
            df = pd.read_csv(path, encoding='utf-8')
            
            stats = {
                "file_path": path,
                "status": "Loaded Successfully",
                "encoding": "utf-8",
                "row_count": len(df),
                "columns": list(df.columns),
                "missing_values": df.isnull().sum().to_dict(),
                "duplicate_rows": int(df.duplicated().sum())
            }
            report["datasets"][name] = stats
            
        except Exception as e:
            report["datasets"][name] = {
                "file_path": path,
                "status": f"Error: {str(e)}"
            }
            
    with open("dataset_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    print("Validation complete. Report saved to dataset_quality_report.json")

if __name__ == "__main__":
    validate_datasets()
