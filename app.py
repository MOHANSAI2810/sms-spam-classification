import sklearn
import streamlit as st
import pickle

# Load the model and vectorizer
model = pickle.load(open('spam.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

# Apply Custom CSS
st.markdown(
    """
    <style>
    /* General Background and Font Styling */
    body {
        background-color: #f0f2f5;
        font-family: 'Arial', sans-serif;
    }
    .main {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    h1, h2 {
        color: #444444;
        text-align: center;
        font-weight: bold;
    }
    .stTextArea textarea {
        background-color: #f9f9f9 !important;
        border-radius: 10px;
        border: 1px solid #cccccc;
    }
    .stButton button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 123, 255, 0.3);
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
    .success-box {
        border-radius: 10px;
        background-color: #dff2e0;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 10px;
    }
    .error-box {
        border-radius: 10px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 10px;
    }
    footer {
        text-align: center;
        margin-top: 50px;
        color: #666;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Content
st.sidebar.title("📧 Email Spam Classifier")
st.sidebar.markdown("""
This app helps you classify emails as either:
- **SPAM**: Unwanted or junk email.
- **HAM**: Legitimate or not spam.
""")
st.sidebar.markdown("Built using **machine learning** technology. 🚀")
st.sidebar.markdown("Developed by: [Mohan](#)")

# Main Content
st.title("📧 Email Spam Classification App")
st.markdown("""
Welcome to the **Email Spam Classification App**!  
Use this tool to predict whether an email is **spam** or **not spam**.  
Simply paste the email content below and click the **Classify Email** button. 🚀  
""")

# Input Text Area
user_input = st.text_area("📨 Paste the Email Text Here", height=150, placeholder="Type or paste the email text here...")

# Classify Button
if st.button("🔍 Classify Email"):
    if user_input:
        data = [user_input]
        vectorized_data = cv.transform(data).toarray()
        result = model.predict(vectorized_data)

        # Display Results
        if result[0] == 0:
            st.markdown('<div class="success-box">✅ The email is **not spam**! You can trust this email. 😊</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">🚫 The email is **spam**! Be cautious about this email. ⚠️</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter email content to classify.")


