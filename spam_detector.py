# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load the dataset (TSV format from UCI)
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])

# Step 3: Convert labels to binary (ham:0, spam:1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 4: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Step 5: Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Predict your own message
def predict_message(msg):
    msg_tfidf = vectorizer.transform([msg])
    prediction = model.predict(msg_tfidf)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Try it
print(predict_message("Get a free recharge now by clicking this link"))
