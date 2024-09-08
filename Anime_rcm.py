import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load and preprocess data
INPUT_DIR = 'C:/Users/anmol/Desktop/VS codes'
anime_csv_path= r'C:\Users\anmol\Desktop\VS codes\anime.csv'
df= pd.read_csv(anime_csv_path)
cols = ["anime_id", "Name", "Genres", "sypnopsis"]
sypnopsis_df = pd.read_csv(INPUT_DIR + '/anime_with_synopsis.csv', usecols=cols)

rating_df = pd.read_csv(INPUT_DIR+ '/rating_complete.csv', usecols=["user_id", "anime_id", "rating"], low_memory=False)
rating_df = rating_df[rating_df['user_id'].isin(rating_df['user_id'].value_counts()[rating_df['user_id'].value_counts() >= 400].index)].copy()
min_rating, max_rating = min(rating_df['rating']), max(rating_df['rating'])
rating_df['rating'] = (rating_df["rating"] - min_rating) / (max_rating - min_rating)

# Remove duplicates
rating_df.drop_duplicates(inplace=True)

# Encode users and anime
user_ids = rating_df["user_id"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
anime_ids = rating_df["anime_id"].unique().tolist()
anime2anime_encoded = {x: i for i, x in enumerate(anime_ids)}
anime_encoded2anime = {i: x for x, i in anime2anime_encoded.items()}
rating_df["user"] = rating_df["user_id"].map(user2user_encoded)
rating_df["anime"] = rating_df["anime_id"].map(anime2anime_encoded)

# Split the data
X = rating_df[['user', 'anime']].values
y = rating_df["rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]


#Function to get Synopsis
def getSypnopsis(anime):
    if isinstance(anime, int):
        return sypnopsis_df[sypnopsis_df.anime_id == anime].sypnopsis.values[0]
    if isinstance(anime, str):
        return sypnopsis_df[sypnopsis_df.Name == anime].sypnopsis.values[0]
    
# Define the model
def RecommenderNet():
    embedding_size = 128
    user = layers.Input(name='user', shape=[1])
    user_embedding = layers.Embedding(name='user_embedding', input_dim=len(user2user_encoded), output_dim=embedding_size)(user)
    anime = layers.Input(name='anime', shape=[1])
    anime_embedding = layers.Embedding(name='anime_embedding', input_dim=len(anime2anime_encoded), output_dim=embedding_size)(anime)
    x = layers.Dot(name='dot_product', normalize=True, axes=2)([user_embedding, anime_embedding])
    x = layers.Flatten()(x)
    x = layers.Dense(1, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("sigmoid")(x)
    model = Model(inputs=[user, anime], outputs=x)
    model.compile(loss='binary_crossentropy', metrics=["mae", "mse"], optimizer='Adam')
    return model

# Train the model
model = RecommenderNet()
checkpoint_filepath = r'C:\Users\anmol\Desktop\VS codes\weights.best.weights.h5'
model_checkpoints = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,  
    monitor='val_loss',      
    mode='min',             
    save_best_only=True      
)
early_stopping = EarlyStopping(patience=3, monitor='val_loss', mode='min', restore_best_weights=True)

history = model.fit(x=X_train_array, y=y_train, batch_size=10000, epochs=20, verbose=1, validation_data=(X_test_array, y_test), callbacks=[model_checkpoints, early_stopping])
model.load_weights(checkpoint_filepath)

def find_similar_animes_by_input(n=10):
    
    user_input = input("Enter the name of the anime: ").strip()
    
    if user_input not in df['English_name'].values:
        print(f"Sorry, '{user_input}' is not found in the dataset.")
        return
    
   
    index = df[df.English_name == user_input].anime_id.values[0]
    
    
    input_genres = df[df.anime_id == index].Genres.values[0]
    
    
    similar_genre_animes = df[(df.Genres.str.contains('|'.join(input_genres.split(',')))) & 
        (~df.English_name.str.contains('unknown', case=False))]
    
    
    encoded_index = anime2anime_encoded.get(index)
    
    
    weights = model.get_layer('anime_embedding').get_weights()[0]
    weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
    
   
    dists = np.dot(weights, weights[encoded_index])
    
    
    similar_anime_ids = similar_genre_animes.anime_id.values
    genre_encoded_indices = [anime2anime_encoded[anime_id] for anime_id in similar_anime_ids if anime_id in anime2anime_encoded]
    
    
    genre_dists = dists[genre_encoded_indices]
    
  
    closest = np.argsort(genre_dists)[-n:]

    
    SimilarityArr = []
    for close in closest:
        decoded_id = anime_encoded2anime.get(genre_encoded_indices[close])
        anime_frame = df[df.anime_id == decoded_id]
        anime_name = anime_frame.English_name.values[0]
        genre = anime_frame.Genres.values[0]
        synopsis= getSypnopsis(decoded_id)
        SimilarityArr.append({"name": anime_name, "genre": genre , "Synopsis":synopsis})

   
    return pd.DataFrame(SimilarityArr)


recommended_animes = find_similar_animes_by_input(n=15)
print(recommended_animes)
