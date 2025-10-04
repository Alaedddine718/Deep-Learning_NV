from __future__ import annotations
import csv, json
from typing import Dict, Any, List

def export_metrics_csv(path: str, rows: List[Dict[str, Any]]):
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def export_metrics_json(path: str, data: Dict[str, Any]):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
