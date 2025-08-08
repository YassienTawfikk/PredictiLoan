import joblib
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold


def get_lr_model():
    return ImbPipeline(steps=[
        ("smote", SMOTE(k_neighbors=5, random_state=42)),
        ("clf", LogisticRegression(
            C=0.1,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000
        ))
    ])


def get_svm_model():
    return ImbPipeline(steps=[
        ("smote", SMOTE(k_neighbors=3, random_state=42)),
        ("clf", SVC(
            C=5,
            kernel="rbf",
            gamma="auto",
            probability=True
        ))
    ])


def save_model(model, path):
    joblib.dump(model, path)


def fit_model(model, features, labels):
    return model.fit(features, labels)


def print_model_accuracy(model, features, labels, name):
    y_pred = model.predict(features)

    acc = accuracy_score(labels, y_pred)
    print(f"{name} Accuracy: {acc * 100:.2f}%")
