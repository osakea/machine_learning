import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from datetime import datetime

# load dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv("test.csv")

print('Jumlah nilai yang hilang di TRAIN:')
print(train_df.isnull().sum())

print("\nJumlah nilai yang hilang di TEST:")
print(test_df.isnull().sum())

def calculate_age(bdate):
    try:
        year=int(bdate.split('.')[-1])
        return datetime.now().year - year
    except:
        return np.nan
for df in [train_df, test_df]:
    df['age'] = df['bdate'].apply(calculate_age)
    df['age'].fillna(df['age'].median(), inplace=True)

# encode data kategorikal
label_cols = ['sex', 'has_photo', 'has_mobile', 'education_form', 'education_status', 'langs', 'life_man', 'people_main', 'city', 'occupation_type', 'occupation_name']

label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    train_df[col]=le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.fit_transform(test_df[col].astype(str))

#pilih fitur dan target
features = ['sex', 'age', 'has_photo', 'has_mobile', 'followers_count', 'graduation', 'education_form', 'relation', 'education_status', 'langs', 'life_main', 'people_main', 'city']

X_train = train_df[features]
X_train = train_df['result']
X_test = test_df[features]

# normalisasi feature
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# melatih model KNN dengan k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

#prediksi pada test.csv
y_pred = knn.predict(X_test)

#simpan hasil prediksi
test_df['result'] = y_pred

