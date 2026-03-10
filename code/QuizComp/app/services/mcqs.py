import os
import pandas as pd
from app.schemas.mcqs import MCQByIdsRequest

def get_mcqs_by_ids(req: MCQByIdsRequest):
    # Prefer the generated MCQs file for this universe
    if req.csv_path:
        csv_path = req.csv_path
    else:
        csv_path = os.path.join("data", str(req.dataUUID), f"mcqs_{req.dataUUID}.csv")

    if not os.path.exists(csv_path):
        raise ValueError(f"MCQ CSV not found: {csv_path}")

    # IMPORTANT: prevent NaN so options don't become missing
    # Auto-detect delimiter (works for "," and ";")
    df = pd.read_csv(csv_path, sep=None, engine="python", keep_default_na=False)

    if "id" not in df.columns:
        df["id"] = df.index

    # Normalize columns (some datasets use different names)
    # Ensure option_a/b/c/d exist (your generated file has them, but this makes it robust)
    for col in ["option_a", "option_b", "option_c", "option_d", "question", "topic_name", "difficulty"]:
        if col not in df.columns:
            df[col] = ""

    ids = [int(x) for x in req.ids]
    sub = df[df["id"].astype(int).isin(ids)].copy()

    items = sub.to_dict(orient="records")

    # Keep requested order + compute missing
    by_id = {int(it["id"]): it for it in items}
    ordered = [by_id[i] for i in ids if i in by_id]
    missing = [i for i in ids if i not in by_id]

    return {"items": ordered, "missing": missing, "csv_used": csv_path}
