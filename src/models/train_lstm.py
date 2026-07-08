import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import os
import torch.nn as nn
import torch.optim as optim

class DemandLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(DemandLSTM, self).__init__()
        self.hidden_size = hidden_size

        # The LSTM Layer : The core memory engine
        # batch_first = True tells it ur data is [Samples, TIme Steps, Features]
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # The Linear Layer: squashes the 50 hidden neurons down to 1 final prediction
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # we only care abt the prediction at the very LAST time step [-1]
        last_time_step_out = lstm_out[:, -1, :]

        # pass the final step through the linear layer to get the prediction
        predictions = self.linear(last_time_step_out)
        return predictions


def prepare_lstm_data():
    # 1. Connect to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}\n")
    
    # 2. Load and Filter Data (Same as Prophet)
    df = pd.read_csv("data/processed/cleaned_sales_data.csv")
    mask = (df['store_id'] == 'STORE_001') & (df['product_id'] == 'PROD_A')
    
    # Extract just the raw numbers into a NumPy array
    sales_data = df[mask]['sales_volume'].values.reshape(-1, 1)
    
    # 3. The Scaling Trap: Squash numbers between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(sales_data)
    
    # 4. Create the Sliding Window (Look back 14 days to predict day 15)
    SEQ_LENGTH = 14
    X, y = [], []
    
    for i in range(len(scaled_data) - SEQ_LENGTH):
        window = scaled_data[i:(i + SEQ_LENGTH)]
        target = scaled_data[i + SEQ_LENGTH]
        X.append(window)
        y.append(target)
        
    X = np.array(X)
    y = np.array(y)
    
    # 5. Convert to PyTorch Tensors and send to GPU
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    
    print("--- TENSOR SHAPES ---")
    print(f"X (Inputs): {X_tensor.shape} -> [Samples, Time Steps, Features]")
    print(f"y (Targets): {y_tensor.shape}")

    print("\nInitializing Deep Learning Model...")

    model = DemandLSTM().to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100

    print("Starting Training Loop...")
    for epoch in range(epochs):
        model.train() # put  the model in training mode
        optimizer.zero_grad() # clear out the old math gradients from the last loop

        # forward pass : ask the model to guess
        y_pred = model(X_tensor)

        # calc how wrong the guess was
        single_loss = loss_function(y_pred, y_tensor)

        # backward pass : send the error back through the network (backpropagation)
        single_loss.backward()
        optimizer.step() # updates the weights to be slightly better next time


        # print progress every 20 epochs
        if epoch % 20 == 0: 
            print(f'Epoch {epoch:3} | Loss: {single_loss.item():.5f}')

    print(f'Final Epoch {epochs} | Loss: {single_loss.item():.5f}')
    print("SUCCESS: LSTM Model Trained Successfully!")

    print("\nStarting 30-Day Future Forecast...")
    model.eval() # put the model in evaluation mode (locks the weights, turns off learning)

    last_window = scaled_data[-SEQ_LENGTH:]
    current_batch = torch.FloatTensor(last_window).view(1, SEQ_LENGTH, 1).to(device)

    future_predictions = []

    # predict 30 days out (autoreggresive loop)
    with torch.no_grad():
        for _ in range(30):
            # predict tomorrow
            next_pred = model(current_batch)
            future_predictions.append(next_pred.item())

            # slide the window : drop the oldest day and add the new prediction at the end
            next_pred_tensor = next_pred.view(1,1,1)
            current_batch = torch.cat((current_batch[:, 1:, :], next_pred_tensor), dim=1)

    # un-scale teh numbers back to real-world sales volume
    future_predictions = np.array(future_predictions).reshape(-1,1)
    real_world_forecast = scaler.inverse_transform(future_predictions)

    # format the dates and save
    last_date = df[mask]['date'].max()
    future_dates = [pd.to_datetime(last_date) + pd.Timedelta(days=1) for i in range(1,31)]

    lstm_df = pd.DataFrame({
        'ds': future_dates,
        'lstm_yhat' : real_world_forecast.flatten()
    })

    # guardrail: no negative sales
    lstm_df['lstm_yhat'] = lstm_df['lstm_yhat'].clip(lower=0)

    # save for the streamlit dashboard
    lstm_df.to_csv("data/processed/lstm_forecast.csv", index=False)
    print("SUCCESS: 30-Day LSTM Forecast saved to data/processed/lstm_forecast.csv")

if __name__ == "__main__":
    prepare_lstm_data()