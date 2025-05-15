import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
    model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
    return tokenizer, model

tokenizer, model = load_model()

st.title("🧠 Empathetic Mental Health Chatbot")
st.write("Hi there. I'm here to support you. You can share anything with me 💙")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", key="input")

if st.button("Send") and user_input:
    history_text = " EOS ".join([f"User: {u} Bot: {b}" for u, b in st.session_state.history])
    input_text = f"Instruction: The user is feeling low or anxious. Respond empathetically. Input: {history_text} EOS User: {user_input} Bot:"
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs, max_length=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.markdown(f"**You:** {user_input}")
    st.markdown(f"**Bot:** {response}")

    st.session_state.history.append((user_input, response))
