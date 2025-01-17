import streamlit as st
import pickle

model = pickle.load(open('spam.pkl','rb'))
cv = pickle.load(open('vectorizer.pkl','rb'))


st.title("Email Spam Classification app")
st.write("This model will prodict whether the email was spam or not")
user_input= st.text_area("Enter an email to classify",height=70)

if st.button("Classify") :
    if user_input:
        data = [user_input]
        vectorized_data = cv.transform(data).toarray()
        result = model.predict(vectorized_data)
        if result[0]==0:
            st.write("The email is not spam")
        else:
            st.write("The email is spam")
    else:
        st.write("Please type Email to classify")
