"""
bridge.py — Sigma-Lab Network Integration Bridge
Connects external data sources (e.g., Skywire) with Sigma analytical engine.
"""

import json
import os
from datetime import datetime
from engine.core import SigmaAnalyzer

class Bridge:
    def __init__(self, data_path="./data/latest/skywire_vitals.json", output_dir="./reports/integrations"):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[Bridge] Loaded data from {self.data_path}")
        return data

    def process(self):
        data = self.load_data()
        metrics = self.extract_metrics(data)
        sigma = SigmaAnalyzer()
        analysis = sigma.evaluate(metrics)
        self.save_output(analysis)
        return analysis

    def extract_metrics(self, data):
        """Extract key Skywire metrics from raw data"""
        payloads = data.get("payloads", [])
        metrics = {
            "node_count": len(payloads),
            "avg_uptime": sum(p.get("uptime", 0) for p in payloads) / max(len(payloads), 1),
            "avg_latency": sum(p.get("latency_ms", 0) for p in payloads) / max(len(payloads), 1),
            "success_ratio": sum(p.get("success_ratio", 0) for p in payloads) / max(len(payloads), 1)
        }
        print(f"[Bridge] Extracted metrics: {metrics}")
        return metrics

    def save_output(self, result):
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = os.path.join(self.output_dir, f"skywire_sigma_analysis_{timestamp}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[Bridge] Analysis saved to {output_file}")


if __name__ == "__main__":
    bridge = Bridge()
    analysis = bridge.process()
    print(f"[Bridge] Final Analysis: {analysis}")
