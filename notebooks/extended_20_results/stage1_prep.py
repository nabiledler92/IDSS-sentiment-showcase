import warnings
warnings.filterwarnings("ignore")
import re, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent          # .../notebooks/extended_20_results
PROJECT_DIR = BASE_DIR.parent.parent                # repo root
DATA_PATH = PROJECT_DIR / "data" / "KopiKenangan_appreviews.xlsx"
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

df = pd.read_excel(DATA_PATH, sheet_name="kopken_dataset")
print("Raw rows:", len(df))
df = df[df['reviewId'] != 'reviewId'].copy()
df['score'] = df['score'].astype(int)
df['content'] = df['content'].astype(str)
df = df[df['content'].str.strip().str.len() > 1]
before = len(df)
df = df.drop_duplicates(subset='content').reset_index(drop=True)
print(f"Rows before dedup: {before}, after dedup: {len(df)}")

def to_label(score):
    if score <= 2: return "Negative"
    elif score == 3: return "Neutral"
    else: return "Positive"
df['sentiment'] = df['score'].apply(to_label)
print(df['sentiment'].value_counts())

stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

t0 = time.time()
df['clean_text'] = df['content'].apply(clean_text)
df['clean_text'] = df['clean_text'].apply(lambda t: stopword_remover.remove(t))
print(f"Preprocessing done in {time.time()-t0:.2f}s")
df = df[df['clean_text'].str.strip().str.len() > 0].reset_index(drop=True)
print("Final dataset size after preprocessing:", len(df))
print(df['sentiment'].value_counts())

le = LabelEncoder()
y = le.fit_transform(df['sentiment'])
print("Label mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['clean_text'], y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("Train size:", len(X_train_text), " Test size:", len(X_test_text))
print("Train class distribution:", np.bincount(y_train))
print("Test class distribution:", np.bincount(y_test))

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)
print("TF-IDF matrix shape (train):", X_train.shape)

min_class_count = np.bincount(y_train).min()
k_neighbors = max(1, min(5, min_class_count - 1))
print("SMOTE k_neighbors used:", k_neighbors)
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Training class distribution after SMOTE:", np.bincount(y_train_res))

joblib.dump({
    "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
    "X_train_res": X_train_res, "y_train_res": y_train_res,
    "tfidf": tfidf, "le": le,
}, CKPT_DIR / "prep.pkl")
print("SAVED checkpoints/prep.pkl")
