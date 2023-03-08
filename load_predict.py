import pandas as pd
import numpy as np
import keras
from gensim.models import word2vec
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model = keras.models.load_model('final_model.h5')

# Load new data
test_data = pd.read_csv("actual_test_data.csv")

# Train the Word2Vec model
w2v = word2vec.Word2Vec(test_data['w2v_embedding'], workers=4, vector_size=200, min_count=5,
                        window=5, sample=1e-3)

# Define function to get tweet embeddings
def get_tweet_embedding(lemmas, model=w2v.wv, embedding_size=200):
    res = np.zeros(embedding_size)
    cnt = 0
    for word in lemmas.split():
        if word in model:
            res += np.array(model[word])
            cnt += 1
    if cnt:
        res = res / cnt
    else:
        res = np.zeros(embedding_size)
    return res

# Get tweet embeddings for test data
test_data['w2v_embedding'] = test_data['w2v_embedding'].apply(get_tweet_embedding)

# Scale the embeddings
scaled_embeddings = np.array(test_data['w2v_embedding'].to_list())
scaler = MinMaxScaler().fit(scaled_embeddings)
scaled_embeddings = scaler.transform(scaled_embeddings)

# Reshape the input data to match the expected shape of the model
scaled_embeddings = np.reshape(scaled_embeddings, (scaled_embeddings.shape[0], scaled_embeddings.shape[1], 1))

# Make predictions on the test data
predictions = model.predict(scaled_embeddings)

# Add the predicted salary as a new column to the test_data dataframe
test_data['salary'] = predictions

# Print the first few rows of the test_data dataframe with the predicted salary
print(test_data[['salary']].head())

# Save the predicted salary values to a CSV file
test_data[['salary']].to_csv('salaries_nn.csv')

# Print the predictions
print(predictions)
