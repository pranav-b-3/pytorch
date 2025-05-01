import subprocess
import itertools
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
# Define alpha and beta ranges to sweep
# alphas = sorted(set([
#     0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.5,
#     1.0, 2.0, 5.0, 7.5, 10.0, 15.0, 20.0
# ]))

# betas = sorted(set([
#     0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.5,
#     1.0, 2.0, 5.0, 7.5, 10.0, 12.5, 15.0
# ]))

alphas = sorted(set([
    0.0,         # no weight on time — only minimize energy
    0.0001,      # barely willing to trade energy
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,         # aggressively sacrificing energy for speed
]))


betas = sorted(set([
    0.0,         # only minimize time
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    25.0,
    50.0,
    75.0,
    100.0,       # extremely latency-tolerant, energy-saving regime
]))


results = []

# Loop over all (alpha, beta) pairs
for alpha, beta in tqdm(itertools.product(alphas, betas), total=len(alphas) * len(betas)):
    env = os.environ.copy()
    env["ALPHA"] = str(alpha)
    env["BETA"] = str(beta)

    try:
        # Run the script and capture output
        proc = subprocess.run(
            ["python", "train_decision_mm.py", "gpu_profile_results.txt", "--heuristic-name", "GPURankingDemo", "--save-dot"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60  # seconds
        )

        # Look for the line that matches: alpha beta correct total
        for line in proc.stdout.splitlines():
            tokens = line.strip().split()
            if len(tokens) == 5:
                try:
                    a, b = float(tokens[0]), float(tokens[1])
                    correct = int(tokens[2])
                    unsure = int(tokens[3])
                    total = int(tokens[4])
                    if a == alpha and b == beta:
                        results.append({
                            "alpha": a,
                            "beta": b,
                            "correct": correct,
                            "unsure": unsure,
                            "total": total,
                            "accuracy": correct / total
                        })
                        break
                    else:
                        print(f"error reading alpha={alpha}, beta={beta}")
                except ValueError:
                    continue
    except subprocess.TimeoutExpired:
        print(f"Timeout: alpha={alpha}, beta={beta}")
        continue

# Create DataFrame
df = pd.DataFrame(results)
print(df)

# Optional: save to CSV
df.to_csv("alpha_beta_accuracy_results_ENERGY.csv", index=False)

