import os
import re
import math
import random
import numpy as np
import glob
import json
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

vectorizer_path = 'vectorizer.pkl'
model_path = 'model.pkl'

if os.path.exists(vectorizer_path) and os.path.exists(model_path):
    print(f'"{vectorizer_path}" and "{model_path}" exists.\n')

else:
    print(f'"{vectorizer_path}" or "{model_path}" does not exists.\n')

    # Load your dataset
    X = []
    y = []
    data_folder = 'Health_Insurance_Claim/*/*/*.json'
    files = glob.glob(data_folder)
    for file in files:
        with open(file, "r") as f:
            jobj = json.load(f)
            inquiry = jobj['inquiry']
            paraphrases = jobj['paraphrases']
            for p in paraphrases:
                X.append(p)
                y.append(inquiry)

    print("Your training data was loaded. (You can skip this process on your service by loading an existing ML model.)\n")

    # Feature Extraction
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    indices = np.array(range(len(X)))
    print('Feature extraction ended. (You can skip this process on your service by loading an existing ML model.)\n')

    print('Generating a vectorizer. This may take a minute. (You can skip this process on your service by loading an existing ML model.)')
    X = tfidf_vectorizer.fit_transform(X)
    pickle.dump(tfidf_vectorizer, open(vectorizer_path, 'wb')) # Save the model for later use
    print(f'Vectorizer was saved in "{vectorizer_path}".\n')

    # Train/test split
    X_train, X_test, y_train, y_test, idx_train, idx_test  = train_test_split(X, y, indices, test_size=0.5, random_state=42)
    # print('Train/test split ended')

    # Model training
    print('Generating an ML model. This may take 10 minutes. (You can skip this process on your service by loading an existing ML model.)')
    model = LogisticRegression()
    model.fit(X_train, y_train)

    pickle.dump(model, open(model_path, 'wb')) # Save the model for later use
    print(f'ML model was saved in "{model_path}".\n')


    print('Testing the accuracy of the ML model generated. (You can skip this process on your service by loading an existing ML model.)')
    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    print(f'Accuracy of the generated ML model: {accuracy}\n')



# Load the vectorizer and the model from the files
with open(vectorizer_path, "rb") as vec_file:
    loaded_vectorizer = pickle.load(vec_file)

with open(model_path, "rb") as model_file:
    loaded_model = pickle.load(model_file)


# Load your test inquiries
inquiries_for_test = []
with open('inquiries', 'r') as f:
	for line in f:
		inquiries_for_test.append(line.strip())


# Prediction
inquiries_vec = loaded_vectorizer.transform(inquiries_for_test)
predicted_categories = loaded_model.predict(inquiries_vec)
print("/* Chatbot's replies to your inquires according to predicted intents */\n")
for i, c in zip(inquiries_for_test, predicted_categories):
	print(f'Your inquiry: \n\t{i}')
	print(f'Chatbot: Your pre-defined answer to the predicted intent: \n\t{c}\n')