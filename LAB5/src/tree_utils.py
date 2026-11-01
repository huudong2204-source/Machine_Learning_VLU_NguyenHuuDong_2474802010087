import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

def export_tree_visualization(model, feature_names, class_names, title, save_filename):
    """Vẽ và lưu sơ đồ cây quyết định."""
    plt.figure(figsize=(15, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        precision=3,
        fontsize=9
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close()

def compute_gini(labels):
    """Tính chỉ số Gini Impurity"""
    arr = np.asarray(labels)
    if len(arr) == 0:
        return 0.0
    _, counts = np.unique(arr, return_counts=True)
    ratios = counts / len(arr)
    return 1.0 - np.sum(ratios ** 2)

def evaluate_best_feature_split(df_features, target_series):
    """Tìm câu hỏi phân chia tối ưu dựa trên Gini Gain"""
    y_arr = target_series.to_numpy()
    total_n = len(y_arr)
    base_gini = compute_gini(y_arr)
    
    max_gain = -1.0
    selected_col = None
    selected_thresh = None

    for col in df_features.columns:
        vals = df_features[col].to_numpy()
        unique_vals = np.sort(np.unique(vals))

        for v1, v2 in zip(unique_vals[:-1], unique_vals[1:]):
            thresh = (float(v1) + float(v2)) / 2.0
            mask = vals <= thresh
            
            y_left, y_right = y_arr[mask], y_arr[~mask]
            w_gini = (len(y_left) / total_n) * compute_gini(y_left) + (len(y_right) / total_n) * compute_gini(y_right)
            gain = base_gini - w_gini

            if gain > max_gain:
                max_gain = gain
                selected_col = col
                selected_thresh = thresh

    return selected_col, selected_thresh, base_gini, base_gini - max_gain
