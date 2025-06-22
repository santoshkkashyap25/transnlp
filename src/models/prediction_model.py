import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, ENSEMBLE_CLASSIFIER_PATH

# Define model/scaler file paths
SCALER_PATH = os.path.join(MODELS_DIR, 'production_scaler.joblib')

# Features used for training
X_COLUMNS = [
    'Culture', 'UK', 'Crimes', 'Situational', 'Immigrants', 'Relationships', 'Politics',
    '0_LDA', '1_LDA', '2_LDA', '3_LDA', '4_LDA', '5_LDA', '6_LDA',
    '0_tfidf', '1_tfidf', '2_tfidf', '3_tfidf', '4_tfidf', '5_tfidf', '6_tfidf'
]

def train_and_save_model(df):
    print("Training new ensemble model...")

    X = df[X_COLUMNS].values
    y = df['rating_type_encoded'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=101, random_state=1)
    rf.fit(X_scaled, y)

    sgd = SGDClassifier(loss='modified_huber', random_state=1)
    sgd.fit(X_scaled, y)

    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=1,
        colsample_bytree=0.3,
        eta=0.05,
        gamma=0.4,
        max_depth=3,
        min_child_weight=1
    )
    xgb.fit(X_scaled, y)

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('sgd', sgd), ('xgb', xgb)],
        voting='soft',
        n_jobs=-1
    )
    ensemble.fit(X_scaled, y)

    joblib.dump(ensemble, ENSEMBLE_CLASSIFIER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Model and scaler saved successfully.")

    return ensemble, scaler

def load_or_train_model():
    if os.path.exists(ENSEMBLE_CLASSIFIER_PATH) and os.path.exists(SCALER_PATH):
        print("Loading existing model and scaler...")
        ensemble = joblib.load(ENSEMBLE_CLASSIFIER_PATH)
        scaler = joblib.load(SCALER_PATH)
        return ensemble, scaler
    else:
        print("Model not found. Training a new one...")
        df_path = os.path.join(PROCESSED_DATA_DIR, 'processed_content_with_clusters.csv')
        df = pd.read_csv(df_path)
        df.dropna(subset=['rating'], inplace=True)

        if 'rating_type_encoded' not in df.columns:
            le = LabelEncoder()
            df['rating_type_encoded'] = le.fit_transform(df['rating_type'])

        # Convert boolean columns to int
        bool_cols = df.select_dtypes(include='bool').columns
        df[bool_cols] = df[bool_cols].astype(int)

        # Ensure cluster columns exist
        if 'cluster_LDA' not in df.columns or 'cluster_tfidf' not in df.columns:
            raise ValueError("cluster_LDA or cluster_tfidf missing from CSV. Cannot create one-hot vectors.")

        # One-hot encode cluster_LDA (0_LDA to 6_LDA)
        lda_df = pd.get_dummies(df['cluster_LDA'], prefix='', prefix_sep='_').reindex(columns=range(7), fill_value=0)
        lda_df.columns = [f"{i}_LDA" for i in range(7)]

        # One-hot encode cluster_tfidf (0_tfidf to 6_tfidf)
        tfidf_df = pd.get_dummies(df['cluster_tfidf'], prefix='', prefix_sep='_').reindex(columns=range(7), fill_value=0)
        tfidf_df.columns = [f"{i}_tfidf" for i in range(7)]

        # Concatenate with df
        df = pd.concat([df, lda_df, tfidf_df], axis=1)

        return train_and_save_model(df)


if __name__ == "__main__":
    model, scaler = load_or_train_model()
