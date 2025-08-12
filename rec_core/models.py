
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
ALS_AVAILABLE=True
try:
    from implicit.als import AlternatingLeastSquares
except Exception:
    ALS_AVAILABLE=False
def build_content_model(items_df: pd.DataFrame):
    text=(items_df["title"].fillna("")+" "+items_df["category"].fillna("")+" "+items_df["description"].fillna(""))
    vectorizer=TfidfVectorizer(min_df=1,max_df=0.9,ngram_range=(1,2),stop_words="english")
    X=vectorizer.fit_transform(text.values); knn=NearestNeighbors(metric="cosine",algorithm="brute"); knn.fit(X)
    id2row={iid:i for i,iid in enumerate(items_df["item_id"].tolist())}; row2id={i:iid for iid,i in id2row.items()}
    return {"vectorizer":vectorizer,"X":X,"knn":knn,"id2row":id2row,"row2id":row2id}
def content_recommend(item_id:str,model,top_k=10):
    if item_id not in model["id2row"]: return []
    idx=model["id2row"][item_id]; vec=model["X"][idx]
    max_neighbors=min(top_k+1, model["X"].shape[0])
    _,indices=model["knn"].kneighbors(vec,n_neighbors=max_neighbors)
    idxs=[int(i) for i in indices.flatten() if int(i)!=idx][:top_k]
    return [model["row2id"][i] for i in idxs]
def build_als_model(ratings_df: pd.DataFrame, factors=70, reg=0.05, iters=20, alpha=15.0, min_user=2, min_item=2, filter_sparsity_fn=None):
    r=filter_sparsity_fn(ratings_df,min_user=min_user,min_item=min_item) if filter_sparsity_fn else ratings_df.copy()
    if len(r)==0: return None
    users=r["user_id"].unique().tolist(); items_list=r["item_id"].unique().tolist()
    u2i={u:i for i,u in enumerate(users)}; it2i={it:i for i,it in enumerate(items_list)}; i2it={i:it for it,i in it2i.items()}
    rows=r["user_id"].map(u2i).values; cols=r["item_id"].map(it2i).values; data=(1.0+alpha*r["rating"].astype(float).values)
    mat=sp.coo_matrix((data,(rows,cols)), shape=(len(users),len(items_list))).tocsr()
    if not ALS_AVAILABLE: return {"fallback_item_item":True,"matrix":mat,"u2i":u2i,"it2i":it2i,"i2it":i2it}
    model=AlternatingLeastSquares(factors=factors,regularization=reg,iterations=iters); model.fit(mat.T)
    return {"model":model,"matrix":mat,"u2i":u2i,"it2i":it2i,"i2it":i2it,"fallback_item_item":False}
def als_recommend(user_id:str, als_pack, top_k=10):
    if als_pack is None or user_id not in als_pack["u2i"]: return []
    uidx=als_pack["u2i"][user_id]; user_row=als_pack["matrix"][uidx]; owned=set(user_row.indices.tolist())
    n_items=als_pack["matrix"].shape[1]; available=max(0, n_items-len(owned))
    if available<=0: return []
    N_eff=min(int(top_k), available)
    if als_pack.get("fallback_item_item"):
        scores=als_pack["matrix"].T @ user_row.T; scores=np.asarray(scores).ravel()
        top=np.argsort(-scores)[: N_eff+len(owned)]; top=[t for t in top if t not in owned][:N_eff]
        return [als_pack["i2it"][t] for t in top]
    try:
        recs=als_pack["model"].recommend(uidx, als_pack["matrix"], N=N_eff, filter_already_liked_items=True)
        return [als_pack["i2it"][i] for i,_ in recs]
    except Exception: return []
def get_popular_items(ratings_df, items_df, top_n=10):
    pop=ratings_df.groupby("item_id")["rating"].sum().reset_index()
    pop=pop.merge(items_df[["item_id","title","category","price"]], on="item_id", how="left")
    return pop.sort_values("rating", ascending=False).head(top_n)["item_id"].tolist()
def hybrid_candidates(content_ids, als_ids, w_als=0.6, cand_k=100):
    scores={}
    for rank,iid in enumerate(content_ids[:cand_k]): scores[iid]=scores.get(iid,0)+(1-w_als)*(1.0/(rank+1))
    for rank,iid in enumerate(als_ids[:cand_k]): scores[iid]=scores.get(iid,0)+w_als*(1.0/(rank+1))
    return [iid for iid,_ in sorted(scores.items(), key=lambda x:-x[1])[:cand_k]]
def mmr_rerank(candidates,K,content_model,lambda_div=0.35):
    if not candidates: return []
    X=content_model["X"]; id2row=content_model["id2row"]; cand=[iid for iid in candidates if iid in id2row]
    if not cand: return candidates[:K]
    rel={iid:1.0/(i+1) for i,iid in enumerate(cand)}; selected=[cand[0]]
    while len(selected)<min(K,len(cand)):
        best_iid,best_score=None,-1e9; sel_rows=[id2row[sid] for sid in selected]
        for iid in cand:
            if iid in selected: continue
            r=rel[iid]; i_row=id2row[iid]
            if sel_rows:
                sims=(X[i_row].dot(X[sel_rows].T)).toarray().ravel(); max_sim=float(sims.max()) if sims.size else 0.0
            else: max_sim=0.0
            score=(1-lambda_div)*r - lambda_div*max_sim
            if score>best_score: best_score, best_iid=score, iid
        if best_iid is None: break
        selected.append(best_iid)
    return selected[:K]
def finalize_rank(content_ids, als_ids, K, w_als, use_mmr, lambda_div, content_model, pool_size=100):
    cand_k=max(pool_size,K) if use_mmr else K
    candidates=hybrid_candidates(content_ids, als_ids, w_als=w_als, cand_k=cand_k)
    return mmr_rerank(candidates, K=K, content_model=content_model, lambda_div=lambda_div) if use_mmr else candidates[:K]
