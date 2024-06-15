import streamlit as st
import torch
import openai
from streamlit_chat import message
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY

# Load model from huggingface
model = BertForSequenceClassification.from_pretrained("Prakarsha01/fine-tuned-legal-bert-v2")
tokenizer = BertTokenizer.from_pretrained("Prakarsha01/fine-tuned-legal-bert-v2")

#Use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function for classification of clause using Legal-BERT
def classify_clause_legal_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

# Function to integrate the output of clause classificationa and risk analysis
def run_gpt_integration(classification_label, risk_analysis, clause):
    prompt = (
        f"Here is a contract clause that has been classified as '{classification_label}':\n\n"
        f"'{clause}'\n\n"
        f"The potential risks identified in this clause are:\n{risk_analysis}\n\n"
    )
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a legal advisor. Please provide an integrated, cohesive explanation of this clause, its classification, and the identified risks. Provide the respone in the following template:"},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

# Define a combined function
def classify_and_analyze_clause(clause):
    classification_result = classify_clause_legal_bert(clause)
    classification_label = "Audit Clause" if classification_result == 1 else "Not an Audit Clause"
    risk_analysis = run_riskAnalysis(clause)
    integrated_response = run_gpt_integration(classification_label, risk_analysis, clause)
    return integrated_response

# Streamlit app
st.title("Contract Clause Classification and Risk Detection")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.messages:
    for i, chat in enumerate(st.session_state.messages):
        message(chat['question'], is_user=True, key=f"user_{i}", avatar_style="big-smile")
        message(chat['answer'], key=f"bot_{i}")
else:
    st.markdown("No chat history yet. Start by entering a contract clause.")

user_input = st.chat_input(placeholder="Enter a contract clause...")

if user_input:
    response = classify_and_analyze_clause(user_input)
    st.session_state.messages.append({"question": user_input, "answer": "\n"+response})
    st.experimental_rerun()
