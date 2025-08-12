
import numpy as np, pandas as pd
from .models import content_recommend, als_recommend, finalize_rank, get_popular_items
def recall_at_k(pred, truth_item, K): return 1.0 if truth_item in pred[:K] else 0.0
def map_at_k(pred, truth_item, K):
    if truth_item in pred[:K]: return 1.0/(pred.index(truth_item)+1)
    return 0.0
def ndcg_at_k(pred, truth_item, K):
    for i,iid in enumerate(pred[:K],start=1):
        if iid==truth_item: return 1.0/np.log2(i+1)
    return 0.0
def list_diversity(item_ids, content_model):
    if len(item_ids)<2: return 0.0
    X=content_model["X"]; id2row=content_model["id2row"]
    rows=[id2row[iid] for iid in item_ids if iid in id2row]
    if len(rows)<2: return 0.0
    ref=rows[0]; sims=(X[ref].dot(X[rows].T)).toarray().ravel()
    if sims.size<=1: return 0.0
    return float(np.mean(1.0 - sims[1:]))
def coverage(all_rec_lists, catalog_size):
    unique=set(); [unique.update(lst) for lst in all_rec_lists]; return len(unique)/max(1,catalog_size)
def novelty(all_rec_lists, item_pop_counts):
    total,n=0.0,0; total_pop=item_pop_counts.sum()
    if total_pop<=0: return 0.0
    for lst in all_rec_lists:
        for iid in lst:
            p=item_pop_counts.get(iid,1.0)/total_pop; total+=-np.log(p+1e-12); n+=1
    return total/max(1,n)
def evaluate_models(ratings_df, items_df, content_model, als_pack, K=10, min_interactions=3,
                    use_mmr=False, lambda_div=0.35, w_als=0.6, pool_size=120):
    per_user=[]; gb=ratings_df.groupby("user_id"); pop_counts=ratings_df.groupby("item_id")["rating"].sum(); rec_lists=[]
    for user, dfu in gb:
        dfu=dfu.sort_values("timestamp")
        if len(dfu)<min_interactions: continue
        train=dfu.iloc[:-1]; test_item=dfu.iloc[-1]["item_id"]
        cand_k=min(max(100,K), items_df.shape[0])
        if len(train)>0: seed=train.iloc[-1]["item_id"]; content_pred=content_recommend(seed, content_model, top_k=cand_k)
        else: content_pred=[]
        als_pred=als_recommend(user, als_pack, top_k=cand_k) if als_pack is not None else []
        if not als_pred and not content_pred: als_pred=get_popular_items(ratings_df, items_df, top_n=K)
        hybrid_pred=finalize_rank(content_pred, als_pred, K=K, w_als=w_als, use_mmr=use_mmr, lambda_div=lambda_div, content_model=content_model, pool_size=pool_size)
        rec_lists.append(hybrid_pred)
        per_user.append({
            "content_recall": recall_at_k(content_pred, test_item, K),
            "als_recall": recall_at_k(als_pred, test_item, K),
            "hybrid_recall": recall_at_k(hybrid_pred, test_item, K),
            "content_map": map_at_k(content_pred, test_item, K),
            "als_map": map_at_k(als_pred, test_item, K),
            "hybrid_map": map_at_k(hybrid_pred, test_item, K),
            "content_ndcg": ndcg_at_k(content_pred, test_item, K),
            "als_ndcg": ndcg_at_k(als_pred, test_item, K),
            "hybrid_ndcg": ndcg_at_k(hybrid_pred, test_item, K),
            "hybrid_diversity": list_diversity(hybrid_pred, content_model),
        })
    if not per_user:
        return pd.DataFrame([{k:0 for k in ["content_recall@K","als_recall@K","hybrid_recall@K","content_map@K","als_map@K","hybrid_map@K","content_ndcg@K","als_ndcg@K","hybrid_ndcg@K","hybrid_diversity","coverage","novelty"]}], index=[f"K={K}"])
    dfm=pd.DataFrame(per_user).mean().to_dict(); cov=coverage(rec_lists, catalog_size=items_df.shape[0]); nov=novelty(rec_lists, pop_counts)
    return pd.DataFrame([{
        "content_recall@K": round(dfm["content_recall"], 4), "als_recall@K": round(dfm["als_recall"], 4), "hybrid_recall@K": round(dfm["hybrid_recall"], 4),
        "content_map@K": round(dfm["content_map"], 4), "als_map@K": round(dfm["als_map"], 4), "hybrid_map@K": round(dfm["hybrid_map"], 4),
        "content_ndcg@K": round(dfm["content_ndcg"], 4), "als_ndcg@K": round(dfm["als_ndcg"], 4), "hybrid_ndcg@K": round(dfm["hybrid_ndcg"], 4),
        "hybrid_diversity": round(dfm["hybrid_diversity"], 4), "coverage": round(cov, 4), "novelty": round(nov, 4),
    }], index=[f"K={K}"])
