# Sentiment-Analysis-of-Social-media-post
📌 Project Overview

Sentiment Analysis is a Natural Language Processing (NLP) technique used to identify and classify emotions expressed in text data.
This project focuses on analyzing social media posts (Twitter data) to determine the sentiment polarity of user opinions.

The model classifies each post into one of the following categories:

Positive

Negative

Neutral

Irrelevant

This system helps organizations and researchers understand public opinion, brand perception, and user feedback on social platforms.

🎯 Objectives

To preprocess raw social media text data

To build a deep learning model for sentiment classification

To analyze emotions expressed in tweets

To evaluate model performance using accuracy and confusion matrix

🗂 Dataset Description

Source: Twitter Sentiment Dataset

Training Data: twitter_training.csv

Validation Data: twitter_validation.csv

Dataset Fields:
Column Name	Description
Tweet_ID	Unique identifier for each tweet
Entity	Topic or brand mentioned
Sentiment	Sentiment label
Tweet_Content	Actual tweet text
🛠 Technologies Used

Programming Language: Python

Libraries & Frameworks:

Pandas & NumPy

NLTK (Text preprocessing)

TensorFlow / Keras

Scikit-learn

Matplotlib

🧹 Text Preprocessing

The following preprocessing steps were applied:

Lowercasing text

Removing URLs, hashtags, mentions

Tokenization

Stopword removal

Lemmatization

Padding sequences for equal length

🧠 Model Architecture

A Bidirectional LSTM (Bi-LSTM) neural network was used:

Embedding Layer

Two Bi-LSTM layers

Dense layer with ReLU activation

Dropout for regularization

Output layer with Softmax activation

This architecture effectively captures contextual dependencies in text.

📈 Model Performance

Evaluation Metric: Accuracy

Final Accuracy: ~ 75%

Loss Function: Sparse Categorical Crossentropy

Optimizer: Adam

A confusion matrix is used to visualize classification performance across sentiment classes.

✅ Results

The model successfully classifies social media posts into sentiment categories with good accuracy.
The performance is suitable for real-world sentiment analysis tasks involving noisy and informal text.

🚀 Future Enhancements

Use transformer models like BERT for higher accuracy

Handle emojis and slang more effectively

Apply data balancing techniques

Deploy the model as a web application

📌 Conclusion

This project demonstrates the effectiveness of deep learning models in analyzing sentiment from social media data.
Despite challenges such as noisy text and ambiguous language, the Bi-LSTM model delivers reliable results and provides valuable insights into public opinion.

👨‍💻 Author

Parveen Kumar
B.Tech Computer Science Engineering
