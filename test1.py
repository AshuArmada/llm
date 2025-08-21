import spacy
from sklearn import svm
import joblib 
# load the large word vector from spacy as lwv
lwv=spacy.load("en_core_web_lg")
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
docs=list(lwv.pipe(train_x))
train_x_wv=[doc.vector for doc in docs]

# train the svm 
svm_wv=svm.SVC(kernel='linear')
svm_wv.fit(train_x_wv,train_y)
#  okk since the loadin and training is done what do we do right now  yes you are wrong  its
# testing 
test_x=["hello does this shop offer which can help me to pass time during my travel"]
test_x_doc=list(lwv.pipe(test_x))
test_x_wv=[dd.vector for dd in test_x_doc]
predict=svm_wv.predict(test_x_wv)
print('prediction = ',predict[0])
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
# import matplotlib.pyplot as plt

# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
# import matplotlib.pyplot as plt

# # Your test data (add more examples for better evaluation)
# test_x = [
#     "hello does this shop offer which can help me to pass time during my travel",
#     "The storyline was really captivating",
#     "These boots fit perfectly and look great"
# ]
# test_y = ["books", "books", "shoes"]  # true labels for your test sentences

# # Convert test sentences to vectors
# test_x_doc = list(lwv.pipe(test_x))
# test_x_wv = [doc.vector for doc in test_x_doc]

# # Predict using SVM
# predictions = svm_wv.predict(test_x_wv)

# # Calculate accuracy
# acc = accuracy_score(test_y, predictions)
# print(f"Test Accuracy: {acc:.2f}")

# # Confusion Matrix
# cm = confusion_matrix(test_y, predictions, labels=svm_wv.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_wv.classes_)
# disp.plot(cmap='Blues')
# plt.title("Confusion Matrix for SVM Text Classifier")
# plt.show()

# # Convert test sentences to vectors
# test_x_doc = list(lwv.pipe(test_x))
# test_x_wv = [doc.vector for doc in test_x_doc]

# # Predict using SVM
# predictions = svm_wv.predict(test_x_wv)

# # Calculate accuracy
# acc = accuracy_score(test_y, predictions)
# print(f"Test Accuracy: {acc:.2f}")

# # Confusion Matrix
# cm = confusion_matrix(test_y, predictions, labels=svm_wv.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_wv.classes_)
# disp.plot(cmap='Blues')
# plt.title("Confusion Matrix for SVM Text Classifier")
# plt.show()
