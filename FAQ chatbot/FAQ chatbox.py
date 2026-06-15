from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faqs = {
    "what is artificial intelligence": "Artificial Intelligence is a technology that allows machines to think and learn like humans.",
    "what is machine learning": "Machine Learning is a part of AI where computers learn from data.",
    "what is python": "Python is a popular programming language used in AI, web development, and automation.",
    "what is data science": "Data Science is the process of collecting, analyzing, and using data to make decisions.",
    "what is deep learning": "Deep Learning is a type of Machine Learning that uses neural networks."
}

questions = list(faqs.keys())

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

print("===== FAQ CHATBOT =====")
print("Ask me questions about AI")
print("Type 'bye' to exit")

while True:
    user_question = input("\nYou: ").lower()

    if user_question == "bye":
        print("Bot: Goodbye!")
        break

    user_vector = vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vector, question_vectors)

    best_match_index = similarity.argmax()
    best_score = similarity[0][best_match_index]

    if best_score > 0.2:
        answer = faqs[questions[best_match_index]]
        print("Bot:", answer)
    else:
        print("Bot: Sorry, I don't know the answer.")