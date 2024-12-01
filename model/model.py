import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib
import json
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

methods2index = {
    "SMS": 0,
    "PayControl": 1,
    "КЭП на токене": 2,
    "КЭП в приложении": 3
}

index2methods = {
    0:"SMS",
    1:"PayControl",
    2:"КЭП на токене",
    3:"КЭП в приложении"
}


def transform_data(data: dict):
    print(type(data))
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
    data = data.copy(deep=True)
    categorical_columns = ["segment", "role", "currentMethod"]
    # label_encoders = {col: LabelEncoder() for col in categorical_columns}

    # Преобразуем категориальные признаки в числа
    for col in categorical_columns:
        if not is_inference:
            label_encoders[col].fit(data[col])  # Мы "обучаем" энкодер на тренировочных данных
        data[col] = label_encoders[col].transform(data[col])

    # Булевы признаки
    data["mobileApp"] = data["mobileApp"].apply(lambda x: 1 if x else 0)

    # Преобразуем все числовые признаки, которые могут иметь разные масштабы
    numeric_columns = ["organizations", "signatures_common_mobile", "signatures_common_web",
                       "signatures_special_mobile", "signatures_special_web", "claims"]
    # scaler = StandardScaler()
    if not is_inference:
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    else:
        data[numeric_columns] = scaler.transform(data[numeric_columns])

    # Преобразуем доступные методы подписи в бинарные признаки
    all_methods = ["SMS", "PayControl", "КЭП на токене", "КЭП в приложении"]
    for method in all_methods:
        data[f"method_{method}"] = data["availableMethods"].apply(lambda x: 1 if method in x else 0)
    # Если это данные для инференса, просто возвращаем признаки X
    X = data.drop(columns=["clientId", "organizationId", "availableMethods"])

    if is_inference:
        # Для инференса целевой переменной нет
        return X

    # Если это данные для тренировки, то мы возвращаем также целевую переменную y
    # Целевая переменная: метод, который будет рекомендован (recommendedMethod)
    y = data[y_col_name].apply(lambda x: methods2index[x])

    # Удаляем из X все ненужные столбцы (которые не являются признаками для модели)
    X = X.drop(columns=[y_col_name])

    return X, y


class Model:
    def __init__(self, model=None):
        if model is None:
            self.model = Pipeline(steps=[
                # ('preprocessor', self.preprocessor),
                ('classifier',
                 RandomForestClassifier(max_depth=20, max_features="sqrt", min_samples_leaf=1, min_samples_split=10,
                                        n_estimators=200, random_state=42))])
        else:
            self.model = model

        categorical_columns = ["segment", "role", "currentMethod"]
        self.label_encoders = {col: LabelEncoder() for col in categorical_columns}

        self.scaler = StandardScaler()

    def train(self, data_train: pd.DataFrame, data_test: pd.DataFrame = None, y_col_name="recommendedMethod",
              test_ratio: int = 0.2, save_path: str = None, encoder_file_path=None, scaler_file_path=None):
        X, y = preprocess_data(data_train, self.label_encoders, self.scaler, y_col_name=y_col_name)
        if data_test is None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        f1_macro = f1_score(y_test, y_pred, average='micro')

        if save_path is not None:
            self.save_model(save_path)

        if encoder_file_path is not None:
            self.save_encoders(encoder_file_path)
        if scaler_file_path is not None:
            self.save_scaler(scaler_file_path)

        return f1_macro

    def hyperparameters_search(self, data_train, y_col_name="recommendedMethod", test_ratio=0.2):
        X, y = preprocess_data(data_train, self.label_encoders, self.scaler, y_col_name=y_col_name)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }

        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1,
                                   scoring='f1_micro')
        grid_search.fit(X_train, y_train)

        print("Лучшие параметры:", grid_search.best_params_)
        print("Лучший F1:", grid_search.best_score_)
        return grid_search.best_params_

    def predict(self, features: pd.DataFrame, use_preprocess=True):
        if use_preprocess:
            data = preprocess_data(features, self.label_encoders, self.scaler, True)
        else:
            data = features
        return self.model.predict(data)

    def save_encoders(self, encoder_file_path='label_encoders.pkl'):
        joblib.dump(self.label_encoders, encoder_file_path)

    def save_scaler(self, scaler_file_path='scaler.pkl'):
        joblib.dump(self.scaler, scaler_file_path)

    def load_encoders(self, encoder_file_path='label_encoders.pkl'):
        self.label_encoders = joblib.load(encoder_file_path)

    def load_scaler(self, scaler_file_path='scaler.pkl'):
        self.scaler = joblib.load(scaler_file_path)

    def save_model(self, file_path: str = "model.pkl"):
        joblib.dump(self.model, file_path)

    def load_model(self, file_path: str = "model.pkl"):
        self.model = joblib.load(file_path)

    def save_all(self, encoder_file_path='label_encoders.pkl', scaler_file_path='scaler.pkl',
                 model_file_path: str = "model.pkl"):
        self.save_model(model_file_path)
        self.save_encoders(encoder_file_path)
        self.save_scaler(scaler_file_path)

    def load_all(self, encoder_file_path='label_encoders.pkl', scaler_file_path='scaler.pkl',
                 model_file_path: str = "model.pkl"):
        self.load_model(model_file_path)
        self.load_encoders(encoder_file_path)
        self.load_scaler(scaler_file_path)


if __name__ == "__main__":
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(probability=True, random_state=42)),
        # ('lr', LogisticRegression(random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42)),
        # ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)),
        # ('catboost', CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, random_state=42, verbose=0))
    ]

    # Мета-модель
    meta_model = LogisticRegression()

    # Создаем стековый классификатор
    stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model)


    data_g = load_data()
    rec_sys = Model(stacking_model)
    # with pd.option_context('display.max_columns', None, 'display.width', None):
    #     print(data_g.head())

    #rec_sys.hyperparameters_search(data_g)
    # Лучшие параметры: {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200}
    # Лучший F1: 0.3926715325503388

    metric = rec_sys.train(data_g)
    rec_sys.save_all()
    print(metric)
    # with pd.option_context('display.max_columns', None, 'display.width', None):
    #     print(data_g.iloc[0:5])
    # print(rec_sys.predict(data_g.drop(columns=["recommendedMethod"]).iloc[0:5]))
    # print(rec_sys.predict(data_g.drop(columns=["recommendedMethod"]).iloc[0:1]))
    #
    # prep_data, y = preprocess_data(data_g, rec_sys.label_encoders, rec_sys.scaler)
    # with pd.option_context('display.max_columns', None, 'display.width', None):
    #     print(prep_data.head())
    #     print(y.head())

    # loaded_model = Model()
    # loaded_model.load_all()
    # print(loaded_model.predict(data_g.drop(columns=["recommendedMethod"])))
