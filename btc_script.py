import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

df = pd.read_csv('5_year_BTC_500.csv', delimiter=',', decimal=',')
df['Дата'] = pd.to_datetime(df['Дата'], format='%d.%m.%Y')
df = df.sort_values('Дата')

data = df['Цена'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

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
X, y = create_sequences(data_normalized, seq_length)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.FloatTensor(y_test).to(device)

class BitcoinDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = BitcoinDataset(X_train, y_train)
test_dataset = BitcoinDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = None
        
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

model = LSTMModel().to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = loss_function(y_pred, batch_y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        test_loss = loss_function(test_preds, y_test)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Train Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}')

model.eval()
with torch.no_grad():
    test_predictions = model(X_test).cpu().numpy()
    test_predictions = scaler.inverse_transform(test_predictions)
    actual_prices = scaler.inverse_transform(y_test.cpu().numpy())

plt.figure(figsize=(14, 6))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(test_predictions, label='Predicted Prices')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

def predict_future(model, data, seq_length, future_days):
    model.eval()
    predictions = []
    last_sequence = data[-seq_length:]
    
    for _ in range(future_days):
        with torch.no_grad():
            input_seq = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
            pred = model(input_seq)
            predictions.append(pred.item())
            last_sequence = np.append(last_sequence[1:], pred.item()).reshape(-1, 1)
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

future_days = 30
future_predictions = predict_future(model, data_normalized, seq_length, future_days)

plt.figure(figsize=(14, 6))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(range(len(actual_prices), len(actual_prices) + future_days), 
         future_predictions, label='Future Predictions')
plt.title('Bitcoin Price Future Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()