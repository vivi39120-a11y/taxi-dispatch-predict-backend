from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"

pred_path = OUT_DIR / "pred_next_hour_advanced.csv"
centroid_path = DATA_DIR / "taxi_zone_centroids.csv"

print("讀取預測檔：", pred_path)
print("讀取 centroid 檔：", centroid_path)

df_pred = pd.read_csv(pred_path)
df_cent = pd.read_csv(centroid_path)[["LocationID", "Borough", "Zone"]]

df = df_pred.merge(
    df_cent,
    left_on="PULocationID",
    right_on="LocationID",
    how="left"
)

df = df.sort_values("pred_rides", ascending=False).reset_index(drop=True)
df["rank_overall"] = df.index + 1

# 全部地區排名
all_rank_path = OUT_DIR / "next_hour_rank_all.csv"
df.to_csv(all_rank_path, index=False, encoding="utf-8-sig")
print("✅ 全區排名：", all_rank_path)

# 前 20 名
top_n = 20
top_rank_path = OUT_DIR / f"next_hour_rank_top{top_n}.csv"
df.head(top_n).to_csv(top_rank_path, index=False, encoding="utf-8-sig")
print(f"✅ 前 {top_n} 名：", top_rank_path)

# 各 Borough 前 5 名
df_borough_top5 = (
    df.sort_values(["Borough", "pred_rides"], ascending=[True, False])
      .groupby("Borough")
      .head(5)
      .reset_index(drop=True)
)

borough_rank_path = OUT_DIR / "next_hour_rank_borough_top5.csv"
df_borough_top5.to_csv(borough_rank_path, index=False, encoding="utf-8-sig")
print("✅ 各 Borough 前 5 名：", borough_rank_path)
