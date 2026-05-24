"""
Monitor Phase 3A Execution Progress
====================================
This script monitors the phase3a_lda_exploration.py execution without blocking.
Run this separately to track progress.
"""

import os
import time
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent / "phase3a_outputs"

def monitor():
    """Monitor phase3a_outputs directory for generated files."""
    print("=" * 60)
    print("  PHASE 3A Execution Monitor")
    print("=" * 60)
    
    last_check_time = time.time()
    k_ranges = [4, 6, 8, 10, 12]
    completed_k_values = set()
    
    while True:
        time.sleep(10)  # Check every 10 seconds
        current_time = time.time()
        elapsed = int(current_time - last_check_time)
        
        # Check what k values have completed
        for k in k_ranges:
            model_file = OUT_DIR / f"lda_model_k{k}.gensim"
            if model_file.exists() and k not in completed_k_values:
                completed_k_values.add(k)
                print(f"\n✓ k={k} training COMPLETED at {time.strftime('%H:%M:%S')}")
        
        # Check for summary
        summary_file = OUT_DIR / "phase3a_summary.json"
        if summary_file.exists():
            print("\n" + "=" * 60)
            print("  PHASE 3A EXECUTION FINISHED!")
            print("=" * 60)
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            print("\nResults Summary:")
            for result in summary.get('results', []):
                print(f"  k={result['k']:2d} | Cv={result['coherence_cv']:.4f} | "
                      f"Barriers={result['barrier_terms_found']:3d} | "
                      f"Topics w/ Barriers={result['unique_topics_with_barriers']}")
            print("\n✓ All artifacts ready in phase3a_outputs/")
            break
        
        # Show progress
        completed = len(completed_k_values)
        print(f"\r[{elapsed:4d}s] Progress: {completed}/{len(k_ranges)} k-values trained | "
              f"Completed: {sorted(completed_k_values)}", end="", flush=True)

if __name__ == "__main__":
    if not OUT_DIR.exists():
        print(f"Waiting for phase3a_outputs to be created...")
        while not OUT_DIR.exists():
            time.sleep(5)
    
    monitor()
