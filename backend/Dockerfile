# Используем официальный образ Python
FROM python:3.11-slim

# Устанавливаем системные зависимости (в том числе libgomp)
RUN apt-get update && apt-get install -y curl \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt /app/

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код приложения
COPY backend/ /app/

COPY model/ /app/model/

# Открываем порт для FastAPI
EXPOSE 8000

# Команда для запуска FastAPI с использованием Uvicorn
CMD ["uvicorn", "back:api", "--host", "0.0.0.0", "--port", "8000", "--reload"]
