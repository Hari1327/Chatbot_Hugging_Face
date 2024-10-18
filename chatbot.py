import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import streamlit as st
import re  # Import regex module to help clean the text

# Hugging Face API token
api_token = "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"

# Hugging Face model repo_id
repo_id = "HuggingFaceH4/starchat-beta"

# Initialize Hugging Face model
llm = HuggingFaceHub(
    repo_id=repo_id,
    huggingfacehub_api_token=api_token,
    model_kwargs={
        "min_length": 30,
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.2,
        "top_k": 50,
        "top_p": 0.95,
        "eos_token_id": 49155
    }
)

# Define the prompt template
prompt = PromptTemplate(template="{myprompt}", input_variables=["myprompt"])

# Set up the LangChain LLM chain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Streamlit app
st.title("AI Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# User input field
user_input = st.text_input("You:", "")

# Define a cleaning function for the model's output
def clean_response(response):
    # Remove any unwanted tokens or characters using regex
    response = re.sub(r'(<\|end\|>|<.*?>)', '', response)  # Remove <|end|> and similar patterns
    response = response.strip()  # Strip any leading or trailing whitespace
    return response

# When the user presses "Send"
if st.button("Send"):
    if user_input:
        # Add the user's input to the chat history
        # st.session_state.chat_history.append(f"You: {user_input}")
        
        # Generate the response using the LLM
        llm_reply = llm_chain.run(myprompt=user_input)
        
        # Clean the response by removing unwanted tokens
        reply = clean_response(llm_reply)
        
        # Add the model's response to the chat history
        st.session_state.chat_history.append(f"AI: {reply}")
        
        # Clear the input box after sending
        user_input = ""

# Display the chat history
for message in st.session_state.chat_history:
    st.text_area("", message, height=100, key=message)
