import random
import json


def generate_data(n_samples=1000, save_path:str="data.json"):
    """
    Генератор данных с добавлением целевой переменной `recommendedMethod`.

    :param n_samples: количество генерируемых строк
    :return: список с генерированными данными в формате словаря
    """

    # Возможные значения для категориальных признаков
    segments = ["Малый бизнес", "Средний бизнес", "Крупный бизнес"]
    roles = ["ЕИО", "Сотрудник"]
    methods = ["SMS", "PayControl", "КЭП на токене", "КЭП в приложении"]

    # Список для хранения данных
    data = []

    # Генерация данных
    for _ in range(n_samples):
        # Генерация случайных данных
        client_id = f"client{random.randint(1000, 9999)}"
        organization_id = f"org{random.randint(1000, 9999)}"
        segment = random.choice(segments)
        role = random.choice(roles)
        organizations = random.randint(1, 300)
        current_method = random.choice(methods)
        mobile_app = random.choice([True, False])
        claims = random.randint(0, 2)

        # Генерация подписанных документов
        signatures_common_mobile = random.randint(0, 20)
        signatures_common_web = random.randint(0, 50)
        signatures_special_mobile = random.randint(0, 10)
        signatures_special_web = random.randint(0, 30)

        # Генерация доступных методов подписи
        available_methods = random.sample(methods, k=random.randint(1, 4))

        # Рекомендация метода подписи (таргет)
        if current_method == "SMS" and claims == 0:
            recommended_method = "PayControl"  # если жалоб нет, рекомендуем более удобный способ
        elif current_method == "PayControl" and len(available_methods) > 2:
            recommended_method = "КЭП на токене"  # если много методов и "PayControl", рекомендуем более безопасный
        else:
            recommended_method = random.choice(methods)  # случайная рекомендация для других случаев

        # Составление строки данных в нужном формате
        row = {
            "clientId": client_id,
            "organizationId": organization_id,
            "segment": segment,
            "role": role,
            "organizations": organizations,
            "currentMethod": current_method,
            "mobileApp": mobile_app,
            "signatures": {
                "common": {
                    "mobile": signatures_common_mobile,
                    "web": signatures_common_web
                },
                "special": {
                    "mobile": signatures_special_mobile,
                    "web": signatures_special_web
                }
            },
            "availableMethods": available_methods,
            "claims": claims,
            "recommendedMethod": recommended_method  # Таргет
        }

        data.append(row)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return data

if __name__ == "__main__":
    data = generate_data(1000)
    # print(df.head())
