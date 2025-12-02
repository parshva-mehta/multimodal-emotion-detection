import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

run_dir = Path("outputs/ravdess_audio_video_baseline")
metrics_path = run_dir / "csv_logs" / "version_0" / "metrics.csv"

df = pd.read_csv(metrics_path)

# Filter out NaNs (Lightning logs some metrics only at epoch end)
df_epoch = df.dropna(subset=["train/loss_epoch", "val/loss"])

plt.figure()
plt.plot(df_epoch["epoch"], df_epoch["train/loss_epoch"], label="train loss")
plt.plot(df_epoch["epoch"], df_epoch["val/loss"], label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("RAVDESS: Train vs Val Loss")
plt.show()

plt.figure()
plt.plot(df_epoch["epoch"], df_epoch["train/acc_epoch"], label="train acc")
plt.plot(df_epoch["epoch"], df_epoch["val/acc"], label="val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.title("RAVDESS: Train vs Val Accuracy")
plt.show()
