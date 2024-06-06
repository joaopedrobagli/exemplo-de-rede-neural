import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 1. Preparar os Dados
texts = ["o gato está", "o gato come", "o cachorro late"]

# Tokenização dos textos
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Preparar dados e rótulos
input_sequences = []
output_words = []

for seq in sequences:
    input_sequences.append(seq[:-1])
    output_words.append(seq[-1])

# Padding das sequências de entrada
maxlen = 2  # Comprimento das sequências de entrada
input_data = pad_sequences(input_sequences, maxlen=maxlen)

# Converter a saída para um array numpy
output_data = np.array(output_words)

print("Dados de entrada:\n", input_data)
print("Dados de saída:\n", output_data)

# 2. Construir o Modelo RNN
model = Sequential()
model.add(Embedding(input_dim=10, output_dim=8, input_length=maxlen))
model.add(SimpleRNN(units=8))
model.add(Dense(units=10, activation='softmax'))  # 10 unidades porque temos um vocabulário pequeno

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Resumo do modelo
model.summary()

# 3. Treinar o Modelo
model.fit(input_data, output_data, epochs=100, verbose=1)

# 4. Fazer Previsões
new_text = "o gato"
new_seq = tokenizer.texts_to_sequences([new_text])
new_input = pad_sequences(new_seq, maxlen=maxlen)

# Fazer previsão
predicted = model.predict(new_input)
predicted_word_index = np.argmax(predicted)

# Converter o índice previsto de volta para a palavra
predicted_word = tokenizer.index_word[predicted_word_index]
print(f"A próxima palavra prevista para '{new_text}' é '{predicted_word}'")