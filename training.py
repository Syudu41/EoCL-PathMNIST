"""Fixed, diverse member configurations. No validation or tuning."""
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SVM_C = [0.1, 1, 3, 10, 30]
RF_PARAMS = [(100, None), (200, None), (100, 20), (200, 20), (300, 30)]


def train_members(features, target, partitions, family, m, seed, log):
    members = {}
    for (j, i), indices in partitions.items():
        if len(np.unique(target[indices])) != 2:
            raise ValueError(f"{family} m={m} bucket={j} member={i}: training partition lacks both targets")
        entry = (1 if family == "SVM" else 0) if m == 1 else i
        if family == "SVM":
            params = dict(C=SVM_C[entry], kernel="rbf", gamma="scale",
                          class_weight="balanced", random_state=seed)
            model = make_pipeline(StandardScaler(), SVC(**params))
        else:
            trees, depth = RF_PARAMS[entry]
            params = dict(n_estimators=trees, max_depth=depth, max_features="sqrt",
                          class_weight="balanced", random_state=seed, n_jobs=-1)
            model = RandomForestClassifier(**params)
        record = dict(classifier=family, m=m, bucket=j, member=i,
                      training_count=len(indices), parameters=params)
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(f"Training {family} m={m} bucket={j} member={i}: {params}", flush=True)
        model.fit(features[indices], target[indices])
        members[j, i] = model
    return members
