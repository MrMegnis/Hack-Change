import React, { useState } from 'react';
import data from './data.json';

const App = () => {
    const [response, setResponse] = useState('');
    const [sentData, setSentData] = useState(''); // Для отображения отправленных данных
    const [receivedData, setReceivedData] = useState(''); // Для отображения полученных данных

    const getRecommendation = async () => {
        // Выбираем случайный элемент из data.json
        const randomIndex = Math.floor(Math.random() * data.length);
        const selectedData = data[randomIndex];

        try {
            // Устанавливаем отправляемые данные для отображения
            setSentData(JSON.stringify(selectedData, null, 2));

            // Отправляем запрос на сервер (POST с JSON в теле)
            const res = await fetch("http://localhost:8000/predict", {
                method: "POST",  // Изменили на POST
                headers: {
                    "Content-Type": "application/json",  // Заголовок для JSON
                },
                body: JSON.stringify(selectedData),  // Данные передаются в теле
            });

            if (!res.ok) {
                throw new Error(`Ошибка при запросе к серверу. Код статуса: ${res.status}`);
            }

            const result = await res.json();

            // Устанавливаем полученные данные для отображения
            setReceivedData(JSON.stringify(result, null, 2));

            // Устанавливаем результат в состояние
            setResponse(result.response || 'Нет данных в поле numbers');
        } catch (error) {
            console.error("Ошибка:", error);

            // Устанавливаем ошибку в состояние
            setResponse('Ошибка при запросе.');
            setReceivedData(error.message); // Отображаем ошибку в поле для ответа
        }
    };

    return (
        <div>
            <h1>Рекомендации</h1>
            <button onClick={getRecommendation}>Получить рекомендацию</button>

            <h2>Ответ сервера:</h2>
            <pre>{response}</pre>

            <h2>Отправленные данные:</h2>
            <pre>{sentData}</pre>

            <h2>Полученные данные:</h2>
            <pre>{receivedData}</pre>
        </div>
    );
};

export default App;
