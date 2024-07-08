import streamlit as st
import requests

st.title("ChatGPT-like Interface with Agent API")

user_input = st.text_input("Enter your question:")
if st.button("Submit"):
    with st.spinner("Processing..."):
        response = requests.post("http://localhost:8000/query", json={"request": user_input})
        data = response.json()
        final_response = data["response"]
        logs = data["logs"]

    st.write("### Response")
    st.write(final_response)

    with st.expander("Show Logs"):
        st.json(logs)