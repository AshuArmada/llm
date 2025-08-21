import spacy
from sklearn import svm
import joblib

# 🔹 Load spaCy model with pretrained word vectors
nlp = spacy.load("en_core_web_lg")

# 🔹 Training data: sentences and labels
train_x = [
    "These sneakers are incredibly comfortable.",
    "I couldn't put that novel down!",
    "The boots are stylish and durable.",
    "The plot of the book was very engaging.",
    "I love the design of these running shoes.",
    "The author's writing style is captivating.",
    "These loafers are perfect for formal events.",
    "The ending of the book was unexpected.",
    "These heels are elegant and comfortable.",
    "The storyline of the novel was unforgettable.",
    "These sandals are perfect for summer.",
    "The book explores deep philosophical themes."
]

train_y = [
    "shoes", "books", "shoes", "books",
    "shoes", "books", "shoes", "books",
    "shoes", "books", "shoes", "books"
]

# 🔹 Vectorize training sentences using spaCy's nlp.pipe (efficient batch processing)
docs = list(nlp.pipe(train_x))
train_x_vectors = [doc.vector for doc in docs]



# 🔹 Train SVM classifier using sentence vectors
svm_wv = svm.SVC(kernel='linear')
svm_wv.fit(train_x_vectors, train_y)

# 🔹 Test prediction on new sentence
test_x = ["the fit is great and comfortable"]
test_docs = list(nlp.pipe(test_x))
test_x_vectors = [doc.vector for doc in test_docs]

# 🔹 Output prediction
prediction = svm_wv.predict(test_x_vectors)
print("Prediction:", prediction[0])
