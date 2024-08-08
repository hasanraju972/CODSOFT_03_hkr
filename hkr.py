import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import Adam

# Define seq_length and num_features
seq_length = 100  # Example sequence length, adjust as needed
num_features = 20  # Example number of unique characters, adjust as needed

# Sample data for demonstration
sample_text = "abcdefghijklmnopqrstuvwxyz"  # Replace with your dataset text
chars = sorted(list(set(sample_text)))
char_to_index = {char: idx for idx, char in enumerate(chars)}
index_to_char = {idx: char for idx, char in enumerate(chars)}

# Example data for training (X_train and y_train need to be properly prepared)
X_train = np.random.rand(1000, seq_length, num_features)  # Replace with actual data
y_train = np.random.randint(0, num_features, (1000, seq_length))  # Replace with actual data

# Define the RNN model
model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, num_features), return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(Dense(num_features))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001))

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

# Function to generate text
def generate_text(model, seed, length):
    result = seed
    for _ in range(length):
        x_pred = np.zeros((1, len(seed), num_features))
        for t, char in enumerate(seed):
            if char in char_to_index:
                x_pred[0, t, char_to_index[char]] = 1.0
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds[-1])
        next_char = index_to_char[next_index]
        result += next_char
        seed = seed[1:] + next_char
    return result

# Example usage
initial_seed = 'initialseedtext'  # Replace with actual seed text
generated_text = generate_text(model, initial_seed, 100)
print(generated_text)
