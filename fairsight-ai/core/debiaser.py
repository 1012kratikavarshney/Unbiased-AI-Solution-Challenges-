import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import (
    ExponentiatedGradient, DemographicParity
)


class Debiaser:

    def reweighting(self, X_train, y_train, sensitive_train):
        weights = np.ones(len(y_train))
        for g in sensitive_train.unique():
            mask = sensitive_train == g
            weights[mask] = (
                len(y_train)
                / (sensitive_train.nunique() * mask.sum())
            )
        model = RandomForestClassifier(
            n_estimators=100, random_state=42)
        model.fit(X_train, y_train, sample_weight=weights)
        return model

    def fairness_constraint(self, X_train, y_train,
                            sensitive_train):
        model = ExponentiatedGradient(
            estimator=LogisticRegression(max_iter=1000),
            constraints=DemographicParity(),
            eps=0.01)
        model.fit(X_train, y_train,
                  sensitive_features=sensitive_train)
        return model
