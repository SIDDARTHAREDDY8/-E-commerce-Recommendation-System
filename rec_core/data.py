
import os, numpy as np, pandas as pd
def load_csv(uploaded_file=None, default_path: str | None = None):
    if uploaded_file is not None: return pd.read_csv(uploaded_file)
    if default_path and os.path.exists(default_path): return pd.read_csv(default_path)
    return None
def _pick(cols_map,*names):
    for n in names:
        if n in cols_map: return cols_map[n]
    return None
def normalize_items(items: pd.DataFrame) -> pd.DataFrame:
    cols={c.lower().strip():c for c in items.columns}; out=pd.DataFrame()
    out["item_id"]=items[_pick(cols,"item_id","asin","asin/isbn")].astype(str)
    out["title"]=items[_pick(cols,"title","name")].astype(str).fillna("")
    out["category"]=items[_pick(cols,"category","categories")].astype(str).fillna("") if _pick(cols,"category","categories") else ""
    out["description"]=items[_pick(cols,"description","desc","keys")].astype(str).fillna("") if _pick(cols,"description","desc","keys") else ""
    price_col=_pick(cols,"price","purchase price per unit","list price per unit")
    if price_col:
        out["price"]=(items[price_col].astype(str).str.replace(r"[^\d\.]", "", regex=True).replace("", np.nan).astype(float))
    else: out["price"]=np.nan
    url_col=_pick(cols,"product_url","url","page_url"); out["product_url"]=items[url_col].astype(str) if url_col else ""
    img_col=_pick(cols,"image_url","image","img"); out["image_url"]=items[img_col].astype(str) if img_col else ""
    return out.drop_duplicates("item_id")
def normalize_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    cols={c.lower().strip():c for c in ratings.columns}; out=pd.DataFrame()
    out["user_id"]=ratings[_pick(cols,"user_id","buyer_name","username")].astype(str)
    out["item_id"]=ratings[_pick(cols,"item_id","asin","asin/isbn")].astype(str)
    out["rating"]=pd.to_numeric(ratings[_pick(cols,"rating","quantity")], errors="coerce").fillna(1.0)
    ts_col=_pick(cols,"timestamp","order_date","reviews.date","time")
    out["timestamp"]=pd.to_datetime(ratings[ts_col], errors="coerce") if ts_col else pd.to_datetime("now")
    out=out.dropna(subset=["user_id","item_id"]).drop_duplicates(subset=["user_id","item_id","timestamp"])
    return out
def filter_sparsity(df,min_user=2,min_item=2,rounds=3):
    cur=df.copy()
    for _ in range(rounds):
        before=len(cur); uc=cur["user_id"].value_counts(); ic=cur["item_id"].value_counts()
        cur=cur[cur["user_id"].isin(uc[uc>=min_user].index)]; cur=cur[cur["item_id"].isin(ic[ic>=min_item].index)]
        if len(cur)==before: break
    return cur
def valid_url(u:str)->bool:
    return isinstance(u,str) and (u.startswith("http://") or u.startswith("https://"))
