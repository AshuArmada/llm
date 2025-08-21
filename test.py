import spacy as sp
import joblib
nlp=sp.load("en_core_web_lg")
svm_model =joblib.load("svm_model.pkl")
def classify_text(text: str)->str:
    doc=nlp(text)
    vector=doc.vector.reshape(1,-1)
    prediction=svm_model.predict(vector)
    return prediction[0]
if __name__=="__main__":
    user_input=input("enter a sentence  about shoes  or boots and to exit enter")

    while user_input!="exit":


     predict= classify_text(user_input)
     print(f"predict catogory :{predict}")
     user_input=input("enter a sentence  about shoes  or boots and to exit enter")

