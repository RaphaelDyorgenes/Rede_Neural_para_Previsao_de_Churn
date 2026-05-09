# Modelos baseline para comparação com a MLP.
import logging

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import SEED

logger = logging.getLogger(__name__)


def get_baselines(seed: int = SEED) -> list[tuple[str, object]]:
    # Retorna lista de (nome, modelo) dos baselines configurados.
    return [
        (
            "Dummy",
            DummyClassifier(strategy="most_frequent", random_state=seed),
        ),
        (
            "Logistic Regression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=seed)),
            ]),
        ),
        (
            "Decision Tree",
            DecisionTreeClassifier(max_depth=5, random_state=seed),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100, max_depth=10,
                random_state=seed, n_jobs=-1,
            ),
        ),
    ]


def get_scoring() -> dict:
    # Retorna dict de métricas para cross_validate.
    return {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1": make_scorer(f1_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score),
    }


def evaluate_baselines(
    X,
    y,
    seed: int = SEED,
    n_splits: int = 5,
) -> list[dict]:
    # Avalia todos os baselines com validação cruzada estratificada.
    # Retorna lista de dicts com nome e métricas de cada modelo.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scoring = get_scoring()
    results = []

    for name, model in get_baselines(seed):
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

        metrics = {
            "name": name,
            "auc_roc": scores["test_roc_auc"].mean(),
            "pr_auc": scores["test_average_precision"].mean(),
            "f1": scores["test_f1"].mean(),
            "precision": scores["test_precision"].mean(),
            "recall": scores["test_recall"].mean(),
        }
        results.append(metrics)

        logger.info(
            "%s — AUC: %.3f | PR-AUC: %.3f | F1: %.3f | Precision: %.3f | Recall: %.3f",
            name, metrics["auc_roc"], metrics["pr_auc"],
            metrics["f1"], metrics["precision"], metrics["recall"],
        )

    return results
