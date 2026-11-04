"""Two deliberately small model families with train-fitted preprocessing."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from credit_risk.data import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS


def build_model(name: str, balanced: bool = False, depth: int = 6) -> Pipeline:
    preprocess = ColumnTransformer([
        ('numeric', StandardScaler() if name == 'logistic' else 'passthrough', NUMERIC_COLUMNS),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLUMNS),
    ], remainder='drop')
    if name == 'logistic':
        estimator = LogisticRegression(C=1.0, class_weight='balanced' if balanced else None,
                                       max_iter=2000, random_state=42)
    elif name == 'forest':
        estimator = RandomForestClassifier(n_estimators=150, max_depth=depth,
            min_samples_leaf=20, random_state=42, n_jobs=1)
    else:
        raise ValueError(f'Unknown model family: {name}')
    return Pipeline([('preprocess', preprocess), ('classifier', estimator)])
