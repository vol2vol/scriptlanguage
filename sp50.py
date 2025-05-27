import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Проверка доступности CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Загрузка данных
data = pd.read_csv('5_year_S&P_500.csv', parse_dates=['Дата'], dayfirst=True)
data['Цена'] = data['Цена'].str.replace('.', '').str.replace(',', '.').astype(float)
data = data.sort_values('Дата')

# Подготовка данных
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Цена'].values.reshape(-1, 1))

# Создание последовательностей для обучения
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Разделение на обучающую и тестовую выборки
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Создание Dataset
class SP500Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SP500Dataset(X_train, y_train)
test_dataset = SP500Dataset(X_test, y_test)

# Параметры обучения
batch_size = 64
learning_rate = 0.001
num_epochs = 250

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Модель LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Обучение модели
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Оценка на тестовых данных
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            test_loss += criterion(outputs, batch_y).item()
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')

# Визуализация потерь
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Прогнозирование на тестовых данных
model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(batch_y.cpu().numpy())

predictions = np.array(predictions).reshape(-1, 1)
actuals = np.array(actuals).reshape(-1, 1)

# Обратное масштабирование данных
predictions = scaler.inverse_transform(predictions)
actuals = scaler.inverse_transform(actuals)

# Визуализация результатов
plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('S&P 500 Price')
plt.title('Actual vs Predicted S&P 500 Prices')
plt.legend()
plt.show()

# Прогнозирование будущих значений
def predict_future(model, last_sequence, future_steps, device):
    model.eval()
    predictions = []
    current_sequence = last_sequence.clone().detach().to(device)  # Исправлено: добавлено .to(device)
    
    with torch.no_grad():
        for _ in range(future_steps):
            input_seq = current_sequence.unsqueeze(0).to(device)
            pred = model(input_seq)
            predictions.append(pred.item())
            
            # Обновляем последовательность, добавляя предсказание и удаляя первый элемент
            current_sequence = torch.cat((current_sequence[1:], pred.view(1, 1)))
    
    return predictions

# Последняя последовательность из тестовых данных (исправлено: добавлено .to(device))
last_sequence = torch.tensor(X_test[-1], dtype=torch.float32).to(device)
future_steps = 30
future_predictions = predict_future(model, last_sequence, future_steps, device)

# Обратное масштабирование предсказаний
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Визуализация будущих предсказаний
plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual Price')
plt.plot(range(len(actuals), len(actuals)+future_steps), future_predictions, label='Future Predictions')
plt.xlabel('Time')
plt.ylabel('S&P 500 Price')
plt.title('Future Predictions of S&P 500 Prices')
plt.legend()
plt.show()