from agent import app

import streamlit as st
import pandas as pd

GROQ_API_KEY = "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"

# Import your app (LangChain/LangGraph graph)
# from your_module import app   # <-- Uncomment and update with your actual module

st.set_page_config(page_title="Sentiment Analysis Agent", layout="centered")
st.title("📊 Sentiment Analysis Agent")

st.write("Enter sentences (one per line) and analyze their sentiment.")

# Let user input their own list
user_texts = st.text_area(
    "Input Text List:",
    placeholder="Type one sentence per line...",
    height=150
)

if st.button("Run Sentiment Analysis"):
    # Convert multi-line input into a list
    input_texts = [t.strip() for t in user_texts.split("\n") if t.strip()]

    if not input_texts:
        st.warning("⚠️ Please enter at least one sentence.")
    else:
        # Run your agent
        final_state = app.invoke({
            "input_texts": input_texts,
            "current_index": 0,
            "sentiment_results": [],
            "accuracy_score": ""
        })

        st.subheader("✅ Final Output")

        # Create table with original texts and results
        df = pd.DataFrame({
            "Text": final_state["input_texts"],
            "Predicted Sentiment": final_state["sentiment_results"],
            "Accuracy":final_state["accuracy_score"]
        })

        st.write("### Sentiment Analysis Results")
        st.dataframe(df, use_container_width=True)

        #st.write("### 📈 Calculated Accuracy Score")
        #st.metric(label="Accuracy", value=final_state["accuracy_score"])


