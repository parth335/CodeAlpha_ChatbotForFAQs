import tkinter as tk
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


nltk.download('stopwords')


faq_data = {
    "What is your return policy?": "You can return any item within 30 days of purchase.",
    "How can I track my order?": "You can track your order from your profile in the 'My Orders' section.",
    "Do you offer international shipping?": "Yes, we offer international shipping to selected countries.",
    "How do I reset my password?": "Click on 'Forgot Password' on the login page and follow the instructions.",
    "What payment methods are accepted?": "We accept credit cards, debit cards, and UPI payments.",
}


tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    tokens = tokenizer.tokenize(text.lower())
    return " ".join([word for word in tokens if word not in stop_words])


questions = list(faq_data.keys())
cleaned_questions = [clean_text(q) for q in questions]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(cleaned_questions)


def get_response(user_question):
    cleaned_input = clean_text(user_question)
    input_vector = vectorizer.transform([cleaned_input])
    similarity = cosine_similarity(input_vector, question_vectors)
    best_match = np.argmax(similarity)

    if similarity[0][best_match] > 0.3:
        return faq_data[questions[best_match]]
    return "Sorry, I couldn't find an answer to that."

# GUI Setup 
root = tk.Tk()
root.title("FAQ Chatbot")
root.geometry("580x600")
root.configure(bg="white")

# Title Label
title = tk.Label(root, text=" FAQ Chatbot", font=("Arial", 18, "bold"), bg="white", fg="#333")
title.pack(pady=10)

# Chat display area
chat_box = tk.Text(root, width=65, height=16, bg="#f9f9f9", padx=10, pady=10, font=("Arial", 11))
chat_box.config(state=tk.DISABLED)
chat_box.pack(pady=(0, 10))

# Display both question and response
def show_chat(question):
    response = get_response(question)
    chat_box.config(state=tk.NORMAL)
    chat_box.insert(tk.END, f"You: {question}\n", "user")
    chat_box.insert(tk.END, f"Bot: {response}\n\n", "bot")
    chat_box.config(state=tk.DISABLED)
    chat_box.yview(tk.END)

# Frame for question buttons
btn_frame = tk.Frame(root, bg="white")
btn_frame.pack(pady=5)

# Create buttons for each FAQ question
for q in questions:
    btn = tk.Button(btn_frame, text=q, command=lambda q=q: show_chat(q),
                    font=("Arial", 10), width=60, bg="#e0e0e0", fg="#333", anchor="w", padx=10, pady=5)
    btn.pack(pady=4)

author_label = tk.Label(root, text="By - Parth Ingole", font=("Arial", 10, "italic"), bg="white", fg="black")
author_label.place(relx=1.0, y=10, anchor='ne', x=-10)


root.mainloop()
