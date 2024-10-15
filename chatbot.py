import streamlit as st
import openai

# Set your OpenAI API key
openai.api_key = 'YOUR_API_KEY'

# Function to interact with the GPT model
def chat_with_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # You can use "gpt-4" if you have access
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

# Streamlit UI
st.title("Chatbot using GPT")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Input box for user message
user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        # Store user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get GPT response
        gpt_response = chat_with_gpt(user_input)
        
        # Store GPT response
        st.session_state.messages.append({"role": "assistant", "content": gpt_response})

# Display chat messages
if st.session_state.messages:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Chatbot:** {message['content']}")
