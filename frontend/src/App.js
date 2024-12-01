import React, { useState } from 'react';
import './App.css'; // Здесь могут быть стили для улучшения внешнего вида
import data from './data.json';

const App = () => {
  const [response, setResponse] = useState(null);
  const [sentData, setSentData] = useState(null);
  const [receivedData, setReceivedData] = useState(null);
  const [userMethods, setUserMethods] = useState(['SMS']); // У пользователя всегда есть SMS
  const [recommendedMethod, setRecommendedMethod] = useState(null);
  const [showRecommendation, setShowRecommendation] = useState(false);

  const methodOptions = [
    { id: 0, name: 'SMS' },
    { id: 1, name: 'PayControl' },
    { id: 2, name: 'КЭП на токене' },
    { id: 3, name: 'КЭП в приложении' },
  ];

  const getRecommendation = async () => {
    const randomIndex = Math.floor(Math.random() * data.length);
    const selectedData = data[randomIndex];

    try {
      setSentData(JSON.stringify(selectedData, null, 2));

      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(selectedData),
      });

      if (!res.ok) {
        throw new Error(`Ошибка при запросе к серверу. Код статуса: ${res.status}`);
      }

      const result = await res.json();

      setReceivedData(JSON.stringify(result, null, 2));
      const recommendedId = result.response;

      // Если метод уже подключен, не нужно показывать рекомендацию
      if (!userMethods.includes(methodOptions[recommendedId].name)) {
        setRecommendedMethod(methodOptions[recommendedId]);
        setShowRecommendation(true);
      } else {
        setRecommendedMethod(null);
        setShowRecommendation(false);
      }

      setResponse('Подписание успешно');
    } catch (error) {
      console.error("Ошибка:", error);
      setResponse('Ошибка при запросе.');
      setReceivedData(error.message);
    }
  };

  const handleConnectMethod = (method) => {
    setUserMethods([...userMethods, method]);
    setRecommendedMethod(null);
    setShowRecommendation(false);
  };

  const handleSetAsPrimary = (method) => {
    // Если метод уже есть, устанавливаем его как основной
    alert(`Метод "${method}" установлен как основной`);
    setRecommendedMethod(null);
    setShowRecommendation(false);
  };

  const handleViewInfo = () => {
    alert(`Информация о методе: ${recommendedMethod.name}`);
  };

  return (
    <div className="app-container">
      <h1>Подпись документа</h1>

      <div className="methods-container">
        <h2>Доступные методы подписания</h2>
        <ul>
          {methodOptions.map((method) => (
            <li key={method.id}>
              <input
                type="checkbox"
                checked={userMethods.includes(method.name)}
                disabled={userMethods.includes(method.name)}
                readOnly
              />
              {method.name}
            </li>
          ))}
        </ul>
      </div>

      <button className="recommendation-button" onClick={getRecommendation}>
        Подписать
      </button>

      {response && (
        <div className="response-container">
          <h2>{response}</h2>

          {showRecommendation && recommendedMethod && (
            <div className="recommendation-box">
              <h3>Рекомендованный метод подписания: {recommendedMethod.name}</h3>
              <button onClick={() => handleConnectMethod(recommendedMethod.name)}>
                Подключить {recommendedMethod.name}
              </button>
              <button onClick={() => handleSetAsPrimary(recommendedMethod.name)}>
                Сделать основным
              </button>
              <button onClick={handleViewInfo}>
                Посмотреть информацию о методе
              </button>
            </div>
          )}
        </div>
      )}

      {sentData && (
        <div className="json-display">
          <h2>Отправленные данные:</h2>
          <pre>{sentData}</pre>
        </div>
      )}

      {receivedData && (
        <div className="json-display">
          <h2>Полученные данные:</h2>
          <pre>{receivedData}</pre>
        </div>
      )}
    </div>
  );
};

export default App;
