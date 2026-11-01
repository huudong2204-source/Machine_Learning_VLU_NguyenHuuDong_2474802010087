import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

def analyze_feature_importances(X_tr, y_tr, X_te, y_te):
    """Phân tích độ quan trọng của thuộc tính qua MDI và Permutation Importance."""
    rng = np.random.default_rng(0)
    
    X_tr_ext = X_tr.copy()
    X_te_ext = X_te.copy()
    X_tr_ext["so_ngau_nhien"] = rng.random(len(X_tr_ext))
    X_te_ext["so_ngau_nhien"] = rng.random(len(X_te_ext))

    rf = RandomForestClassifier(n_estimators=200, random_state=0)
    rf.fit(X_tr_ext, y_tr)

    mdi_scores = pd.Series(rf.feature_importances_, index=X_tr_ext.columns)

    perm_result = permutation_importance(
        rf, X_te_ext, y_te, scoring="accuracy", n_repeats=20, random_state=0
    )

    imp_df = pd.DataFrame({
        "MDI": mdi_scores,
        "permutation_mean": perm_result.importances_mean,
        "permutation_std": perm_result.importances_std
    })

    imp_df["rank_MDI"] = imp_df["MDI"].rank(ascending=False, method="min").astype(int)
    imp_df["rank_permutation"] = imp_df["permutation_mean"].rank(ascending=False, method="min").astype(int)
    
    test_acc = accuracy_score(y_te, rf.predict(X_te_ext))
    return imp_df.sort_values("permutation_mean", ascending=False), test_acc
