
import numpy as np
def content_explanations(seed_id: str, rec_ids: list, content_model, topn=3):
    id2row=content_model["id2row"]; vec=content_model["X"]; vocab=content_model["vectorizer"].get_feature_names_out()
    if seed_id not in id2row: return {rid: [] for rid in rec_ids}
    seed_row=id2row[seed_id]; seed_vec=vec[seed_row].toarray().ravel(); seed_idx=np.argsort(-seed_vec)[:50]
    ex={}
    for rid in rec_ids:
        if rid not in id2row: ex[rid]=[]; continue
        r_vec=vec[id2row[rid]].toarray().ravel(); scores=np.minimum(seed_vec[seed_idx], r_vec[seed_idx])
        top=np.argsort(-scores)[:topn]; ex[rid]=[vocab[seed_idx[i]] for i in top if scores[i]>0]
    return ex
def als_because(user_hist_titles: list, max_items=3):
    return user_hist_titles[-max_items:][::-1] if user_hist_titles else []
