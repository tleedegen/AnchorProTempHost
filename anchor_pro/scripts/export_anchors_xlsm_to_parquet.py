"""
Convert data/raw/equipment.xlsx → data/processed/equipment.parquet
Run locally *or* in GitHub Actions.
"""
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE  = Path("data/raw/anchors.xlsx")
DESTINATION = Path("data/processed/anchors.parquet")

#Reading and cleaning data from excel sheet
anchors_df = pd.read_excel(SOURCE,
                           sheet_name=0,
                           header = 0,
                           usecols = "A:J",
                           thousands = ",",)

# 1st column = anchor_id so we can reference them by name
# anchors_df = anchors_df.set_index(anchors_df.columns[0])

#Convert excel to parquet and save to DESTINATION path.
anchors_df.to_parquet(DESTINATION, index=True)

print(f"✨ Wrote {DESTINATION} with {len(anchors_df):,} rows")
