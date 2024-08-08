import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset (you can expand this with more books)
books = {
    'title': ['Book1', 'Book2', 'Book3', 'Book4'],
    'description': [
        'This is book 1, it is about something interesting.',
        'Book 2 is a novel about a thrilling adventure.',
        'Book 3 talks about science and technology.',
        'This book covers history and ancient civilizations.'
    ]
}

# Create DataFrame
df = pd.DataFrame(books, columns=['title', 'description'])

# Step 2: Text Preprocessing (can be extended for more robust cleaning)
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize text
    filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(filtered_tokens)

df['clean_description'] = df['description'].apply(preprocess_text)

# Step 3: Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_description'])

# Step 4: Calculate Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 5: Recommendation Function
def get_recommendations(title, cosine_sim=cosine_sim, df=df, top_n=5):
    # Get the index of the book that matches the title
    idx = df[df['title'].str.lower() == title.lower()].index[0]

    # Get the pairwise similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top-n most similar books
    sim_scores = sim_scores[1:top_n+1]  # Exclude the first book itself (most similar)

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top-n most similar books
    return df['title'].iloc[book_indices]

# Example usage:
book_title = 'Book1'
recommendations = get_recommendations(book_title)
print(f"Recommendations for {book_title}:")
print(recommendations)