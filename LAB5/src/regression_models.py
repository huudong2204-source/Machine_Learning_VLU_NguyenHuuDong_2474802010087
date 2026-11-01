import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def compare_housing_regression(X_tr, y_tr, X_te, y_te):
    """So sánh DecisionTreeRegressor, RandomForestRegressor và LinearRegression."""
    models = {
        "Cay sau 3": DecisionTreeRegressor(max_depth=3, random_state=0),
        "Duong thang": LinearRegression(),
        "Rung 200 cay": RandomForestRegressor(n_estimators=200, random_state=0)
    }

    new_houses = pd.DataFrame({"dien_tich": [150.0, 250.0]})
    results = []

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        preds_test = model.predict(X_te)
        future_preds = model.predict(new_houses)

        results.append({
            "model": name,
            "MAE_test": round(mean_absolute_error(y_te, preds_test), 4),
            "gia_150m2": round(future_preds[0], 4),
            "gia_250m2": round(future_preds[1], 4)
        })

    rf_model = models["Rung 200 cay"]
    target_250 = pd.DataFrame({"dien_tich": [250.0]})
    tree_predictions = np.array([tree.predict(target_250.to_numpy())[0] for tree in rf_model.estimators_])

    return pd.DataFrame(results), rf_model.predict(target_250)[0], tree_predictions.mean()
