import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import base64

ps = PorterStemmer()


# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Filtering alphanumeric words
    y = [i for i in text if i.isalnum()]

    # Removing stopwords and punctuation, and stemming
    y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    return " ".join(y)


# Load pre-trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('mnb_model.pkl', 'rb'))
# transform_text=pickle.load(open('transform_test.pkl', 'rb'))


# Function to convert image to base64 string
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Load the base64 encoded image
image_path = "spam.jpeg"
image_base64 = image_to_base64(image_path)

# Display the image with specified width and centered alignment
st.write(
    f'<div style="display: flex; justify-content: center;"><img src="data:image/jpeg;base64,{image_base64}" width="250"/></div>',
    unsafe_allow_html=True,
)
# In this corrected code:
st.title("Email/SMS Spam Classifier")

# Add text input area
input_sms = st.text_area("Enter the message")

# Add predict button
if st.button('Predict'):
    # Preprocess text
    transformed_sms = transform_text(input_sms)
    # Vectorize text
    vector_input = tfidf.transform([transformed_sms])
    # Make prediction
    result = model.predict(vector_input)[0]

    # Display result with appropriate message and color
    if result == 1:
        st.error("This is a Spam message.")
        st.balloons()  # Display red balloons for spam
    else:
        st.success("This is NOT a Spam message.")
        st.balloons()  # Display green balloons for non-spam
