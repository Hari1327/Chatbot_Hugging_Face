import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch
# Load the pre-trained model and tokenizer
@st.cache_resource
def load_model():
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Title of the web app
st.title("🤖 Chatbot using Hugging Face")

# Description
st.write("This is a simple chatbot application using the `DialoGPT` model from Hugging Face. Type a message to start a conversation!")

# Initialize session state for chat history
if "chat_history_ids" not in st.session_state:
    st.session_state["chat_history_ids"] = None
if "past_inputs" not in st.session_state:
    st.session_state["past_inputs"] = []

# User input text box
user_input = st.text_input("You: ", "")

# Generate response when user submits a message
if user_input:
    # Tokenize input and append EOS token
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Combine new input with chat history
    bot_input_ids = torch.cat([st.session_state["chat_history_ids"], new_input_ids], dim=-1) if st.session_state["chat_history_ids"] is not None else new_input_ids

    # Generate the bot's response
    st.session_state["chat_history_ids"] = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the response
    response = tokenizer.decode(st.session_state["chat_history_ids"][:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Store the input and response
    st.session_state["past_inputs"].append((user_input, response))

# Display chat history
for user, bot in st.session_state["past_inputs"]:
    st.write(f"You: {user}")
    st.write(f"Chatbot: {bot}")
