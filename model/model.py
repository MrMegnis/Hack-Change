import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import json


def transform_data(data: dict):
    df = pd.DataFrame(data)
    df['signatures_common_mobile'] = df['signatures'].apply(lambda x: x.get('common', {}).get('mobile', 0))
    df['signatures_common_web'] = df['signatures'].apply(lambda x: x.get('common', {}).get('web', 0))
    df['signatures_special_mobile'] = df['signatures'].apply(lambda x: x.get('special', {}).get('mobile', 0))
    df['signatures_special_web'] = df['signatures'].apply(lambda x: x.get('special', {}).get('web', 0))
    df = df.drop(columns=['signatures'])
    return df


def load_data(path: str = "data.json"):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return transform_data(data)


def preprocess_data(data, label_encoders, scaler, is_inference=False, y_col_name: str = "recommendedMethod"):
    """
    Универсальная функция предобработки данных для инференса и тренировки.

    :param data: pd.DataFrame, данные клиента для инференса или тренировки.
    :param is_inference: bool, если True, данные для инференса (целевые переменные нет), иначе для тренировки.
    :return: preprocessed features X и (если не инференс) target y
    """

    categorical_columns = ["segment", "role", "currentMethod"]
    # label_encoders = {col: LabelEncoder() for col in categorical_columns}

    # Преобразуем категориальные признаки в числа
    if not is_inference:
        for col in categorical_columns:
            label_encoders[col].fit(data[col])  # Мы "обучаем" энкодер на тренировочных данных
            data[col] = label_encoders[col].transform(data[col])

    # Булевы признаки
    data["mobileApp"] = data["mobileApp"].apply(lambda x: 1 if x else 0)

    # Преобразуем все числовые признаки, которые могут иметь разные масштабы
    numeric_columns = ["organizations", "signatures_common_mobile", "signatures_common_web",
                       "signatures_special_mobile", "signatures_special_web", "claims"]
    # scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Преобразуем доступные методы подписи в бинарные признаки
    all_methods = ["SMS", "PayControl", "КЭП на токене", "КЭП в приложении"]
    for method in all_methods:
        data[f"method_{method}"] = data["availableMethods"].apply(lambda x: 1 if method in x else 0)
    # Если это данные для инференса, просто возвращаем признаки X
    X = data.drop(columns=["clientId", "organizationId"])

    if is_inference:
        # Для инференса целевой переменной нет
        return X

    # Если это данные для тренировки, то мы возвращаем также целевую переменную y
    # Целевая переменная: метод, который будет рекомендован (recommendedMethod)
    y = data[y_col_name]

    # Удаляем из X все ненужные столбцы (которые не являются признаками для модели)
    X = X.drop(columns=[y_col_name])

    return X, y


class Model:
    def __init__(self, model=None, preprocessor=None):
        if preprocessor is None:
            self.preprocessor = ColumnTransformer([
                ('numerical', 'passthrough', ['organizations', 'claims',
                                              'signatures_common_mobile', 'signatures_common_web',
                                              'signatures_special_mobile', 'signatures_special_web', 'mobileApp'])])
        else:
            self.preprocessor = preprocessor

        if model is None:
            self.model = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
        else:
            self.model = model

        categorical_columns = ["segment", "role", "currentMethod"]
        self.label_encoders = {col: LabelEncoder() for col in categorical_columns}

        self.scaler = StandardScaler()

    def train(self, data_train: pd.DataFrame, data_test: pd.DataFrame = None, y_col_name="currentMethod",
              test_ratio: int = 0.2, save_path: str = None):
        X, y = preprocess_data(data_train, self.label_encoders, self.scaler)
        if data_test is None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        f1_macro = f1_score(y_test, y_pred, average='macro')

        if save_path is not None:
            self.save_model(save_path)

        return f1_macro

    def predict(self, features: pd.DataFrame):
        data = preprocess_data(features, self.label_encoders, self.scaler, True)
        return self.model.predict(data)

    def save_model(self, file_path: str = "model.pkl"):
        joblib.dump(self.model, file_path)

    def load_model(self, file_path: str = "model.pkl"):
        self.model = joblib.load(file_path)


if __name__ == "__main__":
    data_g = load_data()
    rec_sys = Model()
    # with pd.option_context('display.max_columns', None, 'display.width', None):
    #     print(data_g.head())
    prep_data = preprocess_data(data_g, rec_sys.label_encoders, rec_sys.scaler, is_inference=True)
    # with pd.option_context('display.max_columns', None, 'display.width', None):
    #     print(prep_data.head())
    metric = rec_sys.train(data_g)
    print(metric)
    print(data_g.iloc[0:2])
    print(rec_sys.predict(data_g.iloc[0:1]))
