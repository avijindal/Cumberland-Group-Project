import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def chat(user_input):
    return generate_response(user_input)

iface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="Ask a question"),
    outputs="text",
    title="LLM HCI Chatbot",
    description="Course Project Chatbot using FLAN-T5-small"
)

if __name__ == "__main__":
    iface.launch()
