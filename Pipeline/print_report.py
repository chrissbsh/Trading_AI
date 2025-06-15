import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Charger le fichier CSV
file_path = "Pipeline/prediction/predictions_optuna_1_jours.csv"

file_path = "Pipeline/prediction/predictions_optuna_run_20250615_200036_1_jours.csv"
df = pd.read_csv(file_path)

# Extraire les colonnes y_true et y_pred
y_true = df['y_true']
y_pred = df['y_pred']

# Calcul des métriques
accuracy = accuracy_score(y_true, y_pred)
balanced_acc = balanced_accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')
report = classification_report(y_true, y_pred, digits=3)
cm = confusion_matrix(y_true, y_pred)

# Affichage
print(f"   → Accuracy final : {accuracy:.4f}")
print(f"   → Balanced Acc.  : {balanced_acc:.4f}")
print(f"   → F1-macro       : {f1_macro:.4f}")
print(f"   → F1-weighted    : {f1_weighted:.4f}\n")
print(report)
print("Confusion Matrix:")
print(cm)