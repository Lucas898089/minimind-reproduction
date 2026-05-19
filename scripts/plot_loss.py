import re
from pathlib import Path

import matplotlib.pyplot as plt

root = Path.home() / "minimind_project"
logs = {
    "pretrain": root / "results/logs/pretrain_h800_2h.out",
    "sft": root / "results/logs/sft_h800_2h.out",
}

pattern = re.compile(r"Epoch:\[\d+/\d+\]\((\d+)/(\d+)\), loss: ([0-9.]+)")

out_dir = root / "results/loss_curves"
out_dir.mkdir(parents=True, exist_ok=True)

for name, path in logs.items():
    steps = []
    losses = []

    for line in path.read_text(errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            steps.append(int(match.group(1)))
            losses.append(float(match.group(3)))

    print(name, "points:", len(steps))

    csv_path = out_dir / f"{name}_loss.csv"
    csv_path.write_text(
        "step,loss\n" + "\n".join(f"{s},{l}" for s, l in zip(steps, losses))
    )

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"MiniMind {name} loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_loss.png", dpi=180)
    plt.close()
