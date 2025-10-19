import pandas as pd
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from catboost import CatBoostClassifier, Pool

logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')
model = CatBoostClassifier()
model.load_model('./models/my_catboost_new.cbm')
model_th = 0.98
logger.info('Pretrained model imported successfully...')

def make_pred(dt, path_to_file):
    meta_path = Path('./models/model_meta.json')
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        feature_names = meta.get('feature_names', list(dt.columns))
        cat_features = meta.get('cat_features', [])
    else:
        feature_names = list(dt.columns)
        cat_features = dt.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    for col in feature_names:
        if col not in dt.columns:
            dt[col] = np.nan
    X = dt[feature_names]

    if cat_features:
        for c in cat_features:
            if c not in X.columns:
                X[c] = 'cat_NAN'
        X[cat_features] = X[cat_features].astype('string').fillna('cat_NAN')

    pool = Pool(X, cat_features=cat_features, feature_names=feature_names)
    proba = model.predict_proba(pool)[:, 1]

    submission = pd.DataFrame({
        'index': pd.read_csv(path_to_file).index,
        'prediction': (proba > model_th) * 1
    })

    output_dir = Path('./output')
    output_dir.mkdir(exist_ok=True, parents=True)
    submission.to_csv(output_dir / 'sample_submission.csv', index=False)

    importances = model.get_feature_importance(pool)
    fi = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_top5 = fi.sort_values('importance', ascending=False).head(5)
    fi_top5_dict = {row['feature']: float(row['importance']) for _, row in fi_top5.iterrows()}
    with open(output_dir / 'feature_importances_top5.json', 'w', encoding='utf-8') as f:
        json.dump(fi_top5_dict, f, ensure_ascii=False, indent=2)

    plt.figure()
    plt.hist(proba, bins=50, density=True)
    plt.xlabel('Predicted score')
    plt.ylabel('Density')
    plt.title('Predicted score density')
    plt.tight_layout()
    plt.savefig(output_dir / 'scores_density.png')
    plt.close()

    logger.info('Prediction complete for file: %s', path_to_file)
    return submission
