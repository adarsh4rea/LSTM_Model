import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

#Loading the train and test datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")    

# Preprocessing the training data
text_data_train = train_df['Tweets'].astype(str)
labels_train = train_df['label']

label_encoder = LabelEncoder()
labels_encoded_train = label_encoder.fit_transform(labels_train)

max_words = 10000
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(text_data_train)

sequences_train = tokenizer.texts_to_sequences(text_data_train)
max_len = 100  
padded_sequences_train = pad_sequences(sequences_train, maxlen=max_len, padding='post', truncating='post')

#Building the LSTM model
embedding_dim = 16
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
model.add(LSTM(64))
model.add(Dense(3, activation='softmax')) 

#Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#Training the model
model.fit(padded_sequences_train, labels_encoded_train, epochs=10, validation_data=test_df, callbacks=[early_stopping])

#Evaluate the model on the test set
loss, accuracy = model.evaluate(test_df)
print(f'Test Accuracy: {accuracy:.2f}')