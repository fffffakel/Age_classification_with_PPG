import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# Загрузка данных
data = pd.read_csv(r'D:\Proga\AML\PPG_dataset\dataset.csv') 
data.drop(columns=['error'], inplace=True)
data.dropna(inplace=True)

# Выделение признаков и целевой переменной
X = data.drop(['id', 'age_group'], axis=1).values
y = data['age_group'].values - 1 

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Инициализация модели CatBoost
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    verbose=100
)

# Обучение модели
model.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred = model.predict(X_test)

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model on the test set: {accuracy * 100:.2f}%')

##pred ~36%