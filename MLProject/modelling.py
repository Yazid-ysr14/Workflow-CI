import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

train_path = "student_dropout_preprocessing/student_dropout_train_preprocessed.csv"
test_path = "student_dropout_preprocessing/student_dropout_test_preprocessed.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

# Tracking URI sengaja tidak di-set ke server eksternal supaya bisa jalan
# otomatis di lingkungan CI (GitHub Actions) tanpa perlu kredensial tambahan.
# mlflow.set_tracking_uri("http://127.0.0.1:5000/")
# mlflow.set_experiment("Student_Dropout_Classification")

with mlflow.start_run(run_name="ci_run"):
    mlflow.sklearn.autolog()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    print(f"Accuracy: {acc}")
    print(f"F1 Score (macro): {f1}")
