import streamlit as st
import numpy as np
import tensorflow as tf
import tflearn
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import warnings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import speech_recognition as sr
import pyttsx3
from googleapiclient.discovery import build
import pyaudio

warnings.filterwarnings("ignore")
stemmer = LancasterStemmer()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load intents file
with open(r"C:\Users\ADMIN\Downloads\intents.json") as json_data:
    intents = json.load(json_data)

# Process intents
words = []
classes = []
documents = []
ignore_words = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

training = []
output = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Load the model if it exists, otherwise create and train it
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return(np.array(bag))

def classify(sentence):
    ERROR_THRESHOLD = 0.25
    results = model.predict([bow(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list

def response(sentence):
    results = classify(sentence)
    if results:
        while results:
            for intent in intents['intents']:
                if intent['tag'] == results[0][0]:
                    return random.choice(intent['responses'])
            results.pop(0)
    return None

def search_pdf(query, pdf_paths):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=32,
        length_function=len,
    )
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        texts = [text for page in pdf_reader.pages for text in text_splitter.split_text(page.extract_text())]
        for text in texts:
            if query.lower() in text.lower():
                return text
    return None

def search_youtube(query):
    api_key = 'AIzaSyBGos_ev_zgrDxefngfICvWq2taCinLF7Y'  # Replace with your YouTube API key
    youtube = build('youtube', 'v3', developerKey=api_key)

    request = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=1
    )
    response = request.execute()
    
    if response['items']:
        video_id = response['items'][0]['id']['videoId']
        video_title = response['items'][0]['snippet']['title']
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        return video_title, video_url
    return None, None

def chatbot_response(input_data, pdf_paths):
    specific_queries = ["what is", "tell me about", "explain"]
    matched_query = any(input_data.lower().startswith(q) for q in specific_queries)
    
    youtube_result = None
    pdf_json_result = None
    
    if matched_query:
        query_term = input_data.lower()
        for q in specific_queries:
            query_term = query_term.replace(q, '').strip()
        
        # Check YouTube first
        for q in specific_queries:
            if input_data.lower().startswith(q):
                query_term = input_data.lower().replace(q, '').strip()
                video_title, video_url = search_youtube(query_term)
                if video_url:
                    youtube_result = f"Here is a video that might help: {video_title}\n{video_url}"
                    break
        
        # Then check PDFs and JSON
        for q in specific_queries:
            if input_data.lower().startswith(q):
                query_term = input_data.lower().replace(q, '').strip()
                answer_from_pdf = search_pdf(query_term, pdf_paths)
                if answer_from_pdf:
                    pdf_json_result = answer_from_pdf
                else:
                    answer_from_json = response(query_term)
                    if answer_from_json:
                        pdf_json_result = answer_from_json
                break
    
    # Combine or prioritize results
    if youtube_result and pdf_json_result:
        return f"{youtube_result}\n\nAdditional information:\n{pdf_json_result}"
    elif pdf_json_result:
        return pdf_json_result
    else:
        return "Sorry, I don't have an answer for that."

def get_speech_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = None
        try:
            audio = recognizer.listen(source)
            print("Recognizing...")
            query = recognizer.recognize_google(audio)
            print(f"You said: {query}")
            return query
        except sr.UnknownValueError:
            print("Sorry, I did not understand that. Could you please repeat?")
            return None
        except sr.RequestError:
            print("Sorry, my speech service is down. Please try again later.")
            return None
        finally:
            if audio:
                audio = None  # Explicitly set to None to release resources

def chatbot_response_streamlit(input_data, pdf_paths):
    if input_data.lower() == 'exit':
        return "Goodbye!"

    answer = chatbot_response(input_data, pdf_paths)
    return answer

def safe_print_st(answer):
    if answer is not None and answer.strip() != "":
        st.write("Bot:", answer)

def main_streamlit():
    pdf_paths = [
        r"C:\Users\ADMIN\Desktop\unit-2(games).pdf",
        r"C:\Users\ADMIN\Desktop\AI dataset\unit-3(kr).pdf"
    ]

    st.title("EDUCATIONAL ROBOT PLATFORM(AI)")

    mode = st.radio("Would you like to speak or type?", ('speak', 'type'))

    if mode == 'speak':
        st.write("Click the microphone and start speaking...")
        if st.button("Start Listening"):
            input_data = get_speech_input()
            if input_data:
                st.write(f"You said: {input_data}")
                answer = chatbot_response_streamlit(input_data, pdf_paths)
                st.write("Bot:", answer)
    elif mode == 'type':
        input_data = st.text_input("You:")
        if st.button("Submit"):
            if input_data.lower() == 'exit':
                st.write("Goodbye!")
            else:
                answer = chatbot_response_streamlit(input_data, pdf_paths)
                st.write("Bot:", answer)

if __name__ == "__main__":
    main_streamlit()
