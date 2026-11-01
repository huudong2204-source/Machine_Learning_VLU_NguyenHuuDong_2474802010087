import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def evaluate_depth_and_criteria(X_tr, y_tr, X_te, y_te, max_depth_range=range(1, 16)):
    """Đánh giá tác động của max_depth và tiêu chí Gini vs Entropy."""
    records = []
    comparison_summary = []

    for criterion in ["gini", "entropy"]:
        cv_scores_dict = {}

        for depth in max_depth_range:
            clf = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=0)
            mean_cv_acc = cross_val_score(clf, X_tr, y_tr, cv=5, scoring="accuracy").mean()
            
            clf.fit(X_tr, y_tr)
            train_acc = accuracy_score(y_tr, clf.predict(X_tr))

            cv_scores_dict[depth] = mean_cv_acc
            records.append({
                "criterion": criterion,
                "max_depth": depth,
                "accuracy_train": round(train_acc, 4),
                "accuracy_cv": round(mean_cv_acc, 4)
            })

        best_d = max(cv_scores_dict, key=cv_scores_dict.get)
        final_clf = DecisionTreeClassifier(criterion=criterion, max_depth=best_d, random_state=0)
        final_clf.fit(X_tr, y_tr)
        test_acc = accuracy_score(y_te, final_clf.predict(X_te))

        comparison_summary.append({
            "criterion": criterion,
            "best_depth": best_d,
            "accuracy_cv": round(cv_scores_dict[best_d], 4),
            "accuracy_test": round(test_acc, 4)
        })

    return pd.DataFrame(records), pd.DataFrame(comparison_summary)

def compare_pruning_strategies(X_tr, y_tr, X_te, y_te, leaf_candidates=[1, 2, 5, 10, 20, 40]):
    """So sánh cắt tỉa bằng min_samples_leaf vs max_depth cố định."""
    leaf_results = []
    for min_leaf in leaf_candidates:
        clf = DecisionTreeClassifier(min_samples_leaf=min_leaf, random_state=0)
        cv_acc = cross_val_score(clf, X_tr, y_tr, cv=5, scoring="accuracy").mean()
        clf.fit(X_tr, y_tr)
        
        leaf_results.append({
            "min_samples_leaf": min_leaf,
            "depth": clf.get_depth(),
            "n_leaves": clf.get_n_leaves(),
            "accuracy_train": round(accuracy_score(y_tr, clf.predict(X_tr)), 4),
            "accuracy_cv": round(cv_acc, 4)
        })
    
    leaf_df = pd.DataFrame(leaf_results)
    best_leaf_val = int(leaf_df.loc[leaf_df["accuracy_cv"].idxmax(), "min_samples_leaf"])
    
    models = {
        f"min_samples_leaf={best_leaf_val}": DecisionTreeClassifier(min_samples_leaf=best_leaf_val, random_state=0),
        "max_depth=5": DecisionTreeClassifier(max_depth=5, random_state=0)
    }
    
    test_summary = []
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        test_summary.append({
            "model": name,
            "depth": model.get_depth(),
            "n_leaves": model.get_n_leaves(),
            "accuracy_test": round(accuracy_score(y_te, model.predict(X_te)), 4)
        })
        
    return leaf_df, pd.DataFrame(test_summary), best_leaf_val
