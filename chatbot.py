import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, pipeline
import requests
import uuid

# Load BlenderBot model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Load sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

# Function to get weather information
def get_weather(city):
    api_key = "your_openweather_api_key"  # Replace with your OpenWeather API key
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(base_url)
    data = response.json()
    if data["cod"] != "404":
        weather = data["main"]
        return f"The temperature in {city} is {weather['temp']}°C with {data['weather'][0]['description']}."
    else:
        return "City not found."

# Streamlit app setup
st.title("Advanced Chatbot")
st.write("This chatbot can have a basic conversation, provide weather information, and analyze sentiment!")

# Initialize user session
if "user_id" not in st.session_state:
    st.session_state["user_id"] = uuid.uuid4()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = ""

# User input
user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        # Store user input in chat history
        st.session_state["chat_history"] += f"You: {user_input}\n"

        # Analyze sentiment
        sentiment = sentiment_analyzer(user_input)[0]
        st.write(f"Sentiment: {sentiment['label']} with score {sentiment['score']:.2f}")

        # Check for weather intent
        if "weather" in user_input.lower():
            city = user_input.split("in ")[-1]
            weather_info = get_weather(city)
            response = weather_info
        else:
            # Generate response using BlenderBot
            input_ids = tokenizer.encode(st.session_state["chat_history"] + tokenizer.eos_token, return_tensors="pt")
            output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Store response in chat history
        st.session_state["chat_history"] += f"Bot: {response}\n"
        
        # Display conversation history
        st.write(f"**Chatbot:** {response}")

# Show chat history
st.text_area("Chat History", value=st.session_state["chat_history"], height=300)
