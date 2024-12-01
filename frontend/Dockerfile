# Dockerfile для React-фронтенда
FROM node:16

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем package.json и package-lock.json
COPY frontend/package.json /app/
COPY frontend/package-lock.json /app/

# Устанавливаем зависимости
RUN npm install

# Копируем все остальные файлы проекта
COPY frontend/ /app/

# Строим проект
RUN npm run build

# Открываем порт для фронтенда
EXPOSE 3000

# Команда для запуска сервера фронтенда
CMD ["npm", "start"]
