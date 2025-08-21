import matplotlib.pyplot as plt
fig,ax=plt.subplots(figuresize=(10,5))

class category:
    books="books"
    shoes="shoes"

# Comments about books and shoes
train_x = [
    "These sneakers are incredibly comfortable.",
    "I couldn't put that novel down!",
    "The boots are stylish and durable.",
    "The plot of the book was very engaging.",
    "I love the design of these running shoes.",
    "The author's writing style is captivating.",
    "These loafers are perfect for formal events.",
    "The ending of the book was unexpected."
]

train_y = [category.shoes, category.books, category.shoes, category.books, category.shoes, category.books, category.shoes, category.books]


from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer() 
vector_scale_x=vectorizer.fit_transform(train_x)
matrix=vector_scale_x.toarray()
features=vectorizer.get_feature_names_out()

cax=ax.matshow()
from sklearn import svm
classifier=svm.SVC(kernel='linear')
classifier.fit(vector_scale_x, train_y)
test_x=vectorizer.transform([" the story was amazing"])
print(classifier.predict(test_x))
 

# print(vector_scale.toarray()) 
# print(vectorizer.get_feature_names_out())

 