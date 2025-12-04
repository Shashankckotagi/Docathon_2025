import pandas as pd
import numpy as np
import joblib
import os  # Added to handle folder creation
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# --- 0. SETUP ---
print("⏳ Loading Dataset...")
try:
    # Load your specific file
    df = pd.read_csv("authentic_data_2.csv", low_memory=False)
    print(f"✅ Loaded {len(df)} rows.")
except FileNotFoundError:
    print("❌ Error: File 'authentic_data_2.csv' not found.")
    exit()

# --- 1. FEATURE ENGINEERING (Auto-Mapping) ---
print("🧹 Mapping Bacteria Columns...")

def get_col_fuzzy(df, keyword):
    matches = [c for c in df.columns if keyword.lower() in c.lower()]
    if matches:
        print(f"   ✅ Found '{keyword}': {matches[0]}")
        return pd.to_numeric(df[matches[0]], errors='coerce').fillna(0)
    return np.zeros(len(df))

# Extract the 5 Key Biomarkers
df['Total_Firmicutes'] = get_col_fuzzy(df, 'Firmicutes')
df['Total_Bacteroidetes'] = get_col_fuzzy(df, 'Bacteroidetes')
df['Total_Proteobacteria'] = get_col_fuzzy(df, 'Proteobacteria') # Marker for IBD
df['Total_Actinobacteria'] = get_col_fuzzy(df, 'Actinobacteria') # Marker for Gut Health
df['Total_Lactobacillus'] = get_col_fuzzy(df, 'Lactobacillus')   # Marker for Metabolic Health

X = pd.DataFrame({
    'Total_Firmicutes': df['Total_Firmicutes'],
    'Total_Bacteroidetes': df['Total_Bacteroidetes'],
    'Total_Lactobacillus': df['Total_Lactobacillus'], 
    'Total_Escherichia': df['Total_Proteobacteria'],  # Using Proteobacteria as proxy
    'Total_Bifidobacterium': df['Total_Actinobacteria'] # Using Actinobacteria as proxy
})

# --- 2. DEFINE MULTI-CLASS TARGETS ---
print("🏥 Defining Disease Profiles...")

# CLASS 0: Healthy (Default)
df['target'] = 0 

# CLASS 2: IBD (Highest Risk) - Defined by existing diagnosis OR high Proteobacteria
if 'IBD_DIAGNOSIS' in df.columns:
    is_ibd = df['IBD_DIAGNOSIS'].astype(str).str.contains('Crohn|Colitis|IBD', case=False, regex=True)
    df.loc[is_ibd, 'target'] = 2

# Fallback: Top 10% of "Bad Bacteria" (Proteobacteria) = IBD Risk
df.loc[X['Total_Escherichia'] > X['Total_Escherichia'].quantile(0.90), 'target'] = 2

# CLASS 1: IBS / Functional Dysbiosis (High Firmicutes/Bacteroidetes Ratio)
fb_ratio = df['Total_Firmicutes'] / (df['Total_Bacteroidetes'] + 1)
is_ibs = (fb_ratio > fb_ratio.quantile(0.75)) & (df['target'] == 0)
df.loc[is_ibs, 'target'] = 1

# CLASS 3: Metabolic/Diabetes Risk (Very low Lactobacillus)
is_metabolic = (df['Total_Lactobacillus'] < df['Total_Lactobacillus'].quantile(0.15)) & (df['target'] == 0)
df.loc[is_metabolic, 'target'] = 3

print(f"   ✅ Class Distribution: {df['target'].value_counts().to_dict()}")
print("      (0=Healthy, 1=IBS Risk, 2=IBD, 3=Metabolic)")

# --- 3. TRAIN ---
print("🧠 Training Multi-Class XGBoost...")
X_train, X_test, y_train, y_test = train_test_split(X, df['target'], test_size=0.2, random_state=42)

# Removed 'use_label_encoder=False' to fix the UserWarning
model = XGBClassifier(eval_metric='mlogloss', objective='multi:softprob', num_class=4)
model.fit(X_train, y_train)

score = model.score(X_test, y_test) * 100
print(f"🎉 SUCCESS! Model Accuracy: {score:.1f}%")

# --- 4. SAVE MODEL ---
save_folder = "ml_model"
file_name = "gut_model.pkl"

# Check if folder exists, if not create it
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"📂 Created directory: {save_folder}")

# Save the model inside the folder
full_path = os.path.join(save_folder, file_name)
joblib.dump(model, full_path)
print(f"💾 Model saved successfully to: {full_path}")