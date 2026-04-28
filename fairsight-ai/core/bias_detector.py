import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame
)


class BiasDetector:

    def analyze_dataset(self, df, sensitive_col, target_col):
        rates = df.groupby(sensitive_col)[target_col].mean()
        vals = list(rates.values)
        dir_ratio = min(vals) / max(vals) if max(vals) > 0 else 0
        gap = max(vals) - min(vals)
        return {
            'group_rates': rates.to_dict(),
            'disparate_impact_ratio': round(dir_ratio, 3),
            'selection_gap': round(gap, 3),
            'status': (
                'CRITICAL' if dir_ratio < 0.7
                else 'HIGH' if dir_ratio < 0.8
                else 'MODERATE' if dir_ratio < 0.9
                else 'FAIR'
            ),
            'most_favored': rates.idxmax(),
            'least_favored': rates.idxmin()
        }

    def analyze_model(self, y_true, y_pred,
                      sensitive_features, feature_name):
        dpd = abs(demographic_parity_difference(
            y_true, y_pred,
            sensitive_features=sensitive_features))
        eod = abs(equalized_odds_difference(
            y_true, y_pred,
            sensitive_features=sensitive_features))
        mf = MetricFrame(
            metrics=accuracy_score,
            y_true=y_true, y_pred=y_pred,
            sensitive_features=sensitive_features)
        group_acc = mf.by_group.to_dict()
        acc_gap = max(group_acc.values()) - min(group_acc.values())
        penalty = (
            min(dpd * 200, 40)
            + min(eod * 200, 40)
            + min(acc_gap * 100, 20)
        )
        score = round(max(0, 100 - penalty), 1)
        return {
            'feature': feature_name,
            'fairness_score': score,
            'demographic_parity_difference': round(dpd, 4),
            'equalized_odds_difference': round(eod, 4),
            'accuracy_gap': round(acc_gap, 4),
            'group_accuracies': {
                str(k): round(v, 4) for k, v in group_acc.items()
            },
            'dpd_status': (
                'HIGH' if dpd > 0.2
                else 'MODERATE' if dpd > 0.1
                else 'FAIR'
            ),
            'eod_status': (
                'HIGH' if eod > 0.2
                else 'MODERATE' if eod > 0.1
                else 'FAIR'
            )
        }
