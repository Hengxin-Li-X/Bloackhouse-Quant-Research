from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def best_ofi(row: pd.Series, lvl: int, side: str) -> int:

    px = row[f"{side}_px_{lvl:02d}"]
    sz = row[f"{side}_sz_{lvl:02d}"]
    px_prev = row[f"prev_{side}_px_{lvl:02d}"]
    sz_prev = row[f"prev_{side}_sz_{lvl:02d}"]

    if pd.isna(px_prev):
        return 0

    if side == "bid":
        if px > px_prev:
            return int(sz)
        elif px == px_prev:
            return int(sz - sz_prev)
        else:
            return -int(sz_prev)
    else:
        if px < px_prev:
            return int(sz_prev)
        elif px == px_prev:
            return -int(sz - sz_prev)
        else:
            return -int(sz)


def Mulit_Integrated_ofi(df: pd.DataFrame) -> pd.DataFrame:
    lvls = range(10)

    for side in ("bid", "ask"):
        for lvl in lvls:
            for fld in ("px", "sz"):
                col = f"{side}_{fld}_{lvl:02d}"
                df[f"prev_{col}"] = df[col].shift(1)

    for lvl in lvls:
        df[f"ofi_lvl{lvl}"] = (
            df.apply(lambda r: best_ofi(r, lvl, "bid"), axis=1)
            + df.apply(lambda r: best_ofi(r, lvl, "ask"), axis=1)
        )
        df[f"depth_lvl{lvl}"] = df[f"bid_sz_{lvl:02d}"] + df[f"ask_sz_{lvl:02d}"]

    df["bucket"] = pd.to_datetime(df["ts_event"]).dt.floor("1min")
    agg_dict = {f"ofi_lvl{lvl}": "sum" for lvl in lvls}
    agg_dict |= {f"depth_lvl{lvl}": "mean" for lvl in lvls}
    minute = df.groupby("bucket").agg(agg_dict)

    for lvl in lvls:
        depth = minute[f"depth_lvl{lvl}"].replace(0, np.nan)
        minute[f"ofi_lvl{lvl}"] = (minute[f"ofi_lvl{lvl}"] / depth).fillna(0)
        minute.drop(columns=f"depth_lvl{lvl}", inplace=True)

    ofi_matrix = minute[[f"ofi_lvl{lvl}" for lvl in lvls]].to_numpy()
    pca = PCA(n_components=1)
    weights = pca.fit(ofi_matrix).components_[0]
    weights /= np.sum(np.abs(weights))
    minute["integrated_ofi"] = ofi_matrix @ weights

    minute.rename(columns={"ofi_lvl0": "best_ofi"}, inplace=True)

    ordered_cols: List[str] = [
        "best_ofi",
        *[f"ofi_lvl{lvl}" for lvl in range(1, 10)],
        "integrated_ofi",
    ]
    return minute[ordered_cols].reset_index()


def cross_asset_ofi(ofi_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_minutes = sorted({idx for df in ofi_dict.values() for idx in df["bucket"]})
    aligned = []
    for sym, mdf in ofi_dict.items():
        mdf = mdf.set_index("bucket").reindex(all_minutes).fillna(0)
        mdf.columns = pd.MultiIndex.from_product([[sym], mdf.columns])
        aligned.append(mdf)
    return pd.concat(aligned, axis=1).sort_index()


if __name__ == "__main__":

    INPUT_FILE = Path("first_25000_rows.csv")
    raw = pd.read_csv(INPUT_FILE).sort_values("ts_event").reset_index(drop=True)
    ofi_minutes = Mulit_Integrated_ofi(raw)

    symbol = raw["symbol"].iloc[0]
    OUT_FILE = INPUT_FILE.with_name(f"{symbol}_ofi.csv")
    ofi_minutes.to_csv(OUT_FILE, index=False)
    print("Done")
