import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# CSS
# -----------------------------
custom_css = """
.gradio-container {
  background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 40%, #f8fafc 100%);
  min-height: 100vh;
  padding: 20px !important;
}

.agent-shell {
  background: rgba(255, 255, 255, 0.85);
  border: 1px solid rgba(99,102,241,0.15);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(99,102,241,0.15);
  padding: 16px;
  max-width: 900px;
  margin: auto;
}

.agent-title h1 {
  margin: 0 !important;
  color: #1e293b;
  font-size: 22px !important;
}

.agent-title p {
  margin: 4px 0 0 0 !important;
  color: #475569;
  font-size: 13px !important;
}

.agent-status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 10px;
  border-radius: 999px;
  background: rgba(34,197,94,0.1);
  color: #166534;
  font-size: 12px;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #22c55e;
}

.chat-card {
  background: #ffffff;
  border: 1px solid rgba(148,163,184,0.2);
  border-radius: 12px;
  padding: 10px;
}

.chat-card .message {
  font-size: 13px !important;
  padding: 8px 12px !important;
}

.input-row {
  margin-top: 12px;
  background: #f1f5f9;
  border-radius: 999px;
  padding: 8px 12px;
  border: 1px solid rgba(148,163,184,0.25);
}

.input-row textarea,
.input-row input {
  background: transparent !important;
  color: #0f172a !important;
  font-size: 14px !important;
  caret-color: #6366f1 !important;
  border: none !important;
}

.input-row textarea::placeholder,
.input-row input::placeholder {
  color: #64748b !important;
}

.primary-glow button {
  background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
  color: white !important;
  border-radius: 999px !important;
}

.reset-soft button {
  background: #e2e8f0 !important;
  color: #1e293b !important;
  border-radius: 999px !important;
}

footer {
  display: none !important;
}
"""


# -----------------------------
# Model loading
# -----------------------------
MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def generate_response(user_prompt: str) -> str:
    # Instruction prompt helps FLAN-T5 produce better answers
    prompt = f"Answer clearly and briefly:\n{user_prompt}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=160)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response or "Sorry — I couldn't generate a response. Please rephrase your question."


def user_submit(user_message, history):
    if history is None:
        history = []
    if not user_message or not user_message.strip():
        return "", history
    history = history + [{"role": "user", "content": user_message.strip()}]
    return "", history


def bot_response(history):
    if not history:
        return history
    last_user = history[-1]["content"]
    reply = generate_response(last_user)
    history = history + [{"role": "assistant", "content": reply}]
    return history


def reset_all():
    return [], ""


def build_demo():
    with gr.Blocks(title="AI Chat Agent", theme=gr.themes.Soft(), css=custom_css) as demo:
        with gr.Column(elem_classes="agent-shell"):
            # Header row
            with gr.Row(elem_classes="agent-header"):
                with gr.Column(elem_classes="agent-title", scale=8):
                    gr.Markdown(
                        """
                        # ✨ LLM HCI Chatbot
                        Course Project Chatbot using FLAN-T5-small.
                        """
                    )
                with gr.Column(scale=2, min_width=160):
                    gr.HTML(
                        '<div class="agent-status"><span class="dot"></span><span>Online</span></div>'
                    )

            # Chat
            with gr.Column(elem_classes="chat-card"):
                chatbot = gr.Chatbot(height=520, show_label=False)
                state = gr.State([])

            # Input + buttons row
            with gr.Row(elem_classes="input-row"):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Message Nova…",
                    lines=1,
                    autofocus=True,
                    scale=8,
                    container=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, elem_classes="primary-glow")
                reset_btn = gr.Button("Reset", variant="secondary", scale=1, elem_classes="reset-soft")

            # Enter key
            txt.submit(
                fn=user_submit,
                inputs=[txt, state],
                outputs=[txt, state],
                queue=False,
            ).then(
                fn=bot_response,
                inputs=[state],
                outputs=[state],
                queue=False,
            ).then(
                fn=lambda h: h,
                inputs=[state],
                outputs=[chatbot],
                queue=False,
            )

            # Send click
            send_btn.click(
                fn=user_submit,
                inputs=[txt, state],
                outputs=[txt, state],
                queue=False,
            ).then(
                fn=bot_response,
                inputs=[state],
                outputs=[state],
                queue=False,
            ).then(
                fn=lambda h: h,
                inputs=[state],
                outputs=[chatbot],
                queue=False,
            )

            # Reset
            reset_btn.click(
                fn=reset_all,
                inputs=None,
                outputs=[state, txt],
                queue=False,
            ).then(
                fn=lambda: [],
                inputs=None,
                outputs=[chatbot],
                queue=False,
            )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
