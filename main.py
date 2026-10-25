import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg') #precisei colocar para rodar na venv !!

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix



def flatten_text(X):
    """Garante que o TfidfVectorizer receba uma Série 1-D de strings."""
    if hasattr(X, "iloc"):                 # veio como DataFrame
        s = X.iloc[:, 0]
    else:                                   # veio como array (n, 1) ou (n,)
        arr = np.asarray(X)
        s = pd.Series(arr.ravel() if arr.ndim > 1 else arr)
    return s.fillna("").astype(str)

df = pd.read_csv('Itinerarios_de_Servicos_de_Onibus_Regulares.csv')


# Classificação de consorcio em função da extensão, a tarifa, o tipo da rota, a direção e o destino.

# Preparação dos dados para classificação

for col in ["extensao", "tarifas"]:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.strip().replace({"": np.nan, "-": np.nan}),
        errors="coerce",
    )

# --- direcao: 0/1 → string (OneHotEncoder aceita) ---
df["direcao"] = df["direcao"].astype(str)

# --- descartar linhas sem target ---
df = df.dropna(subset=["consorcio"]).reset_index(drop=True)

# Preparação de y e x

y = df["consorcio"]

numericas   = ["extensao", "tarifas"]
categoricas = ["tipo_rota", "direcao"]
textuais    = ["destino"]

X = df[numericas + categoricas + textuais].copy()

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]), numericas),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ]), categoricas),

    ("txt", Pipeline([
        ("flatten", FunctionTransformer(flatten_text)),
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,     # inteiro (e não mais 2)
            max_df=1.0,   # inteiro-compatível: sem corte de termos comuns
        )),
    ]), textuais),
], sparse_threshold=0.3)

# Random Forest Classifier - baseado em árvores de decisão 

model = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    model, X, y, cv=cv, scoring="f1_macro",
    n_jobs=-1, error_score="raise",   # ← mostra o traceback real se falhar
)

# Treino e avaliação

# O treino do modelo foi feito com 75% dos dados enquanto 25% foi reservado para testar ele

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, zero_division=0))

labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(
    pd.DataFrame(cm, index=labels, columns=labels),
    annot=True, fmt="d", cmap="Blues",
)
plt.ylabel("real")
plt.xlabel("predito")
plt.title("Matriz de confusão — consórcio")
plt.tight_layout()
plt.savefig("confusao_consorcio.png")