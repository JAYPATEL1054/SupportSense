"""
SupportSense demo — Gradio interface.

Expects two folders alongside this script (or update the paths below):
  ./department_model/
  ./urgency_model/

These are produced by running the training notebook (notebook/SupportSense_Training.ipynb),
which saves them via trainer.save_model(...) + tokenizer.save_pretrained(...).

If you've uploaded the models to the Hugging Face Hub instead, replace the paths
below with the Hub repo IDs, e.g. "yourusername/supportsense-department".
"""

from transformers import pipeline
import gradio as gr

DEPARTMENT_MODEL_PATH = "./department_model"
URGENCY_MODEL_PATH = "./urgency_model"

department_pipe = pipeline("text-classification", model=DEPARTMENT_MODEL_PATH, tokenizer=DEPARTMENT_MODEL_PATH)
urgency_pipe = pipeline("text-classification", model=URGENCY_MODEL_PATH, tokenizer=URGENCY_MODEL_PATH)

# Must match the label order used during training (alphabetical, from sorted(df[col].unique()))
department_labels = ["Account", "Billing", "General", "Technical"]
id2dept = {idx: label for idx, label in enumerate(department_labels)}

urgency_labels = ["High", "Low", "Medium"]
id2urg = {idx: label for idx, label in enumerate(urgency_labels)}


def predict_ticket(text):
    dept_result = department_pipe(text)[0]
    urg_result = urgency_pipe(text)[0]

    dept_id = int(dept_result["label"].split("_")[-1])
    urg_id = int(urg_result["label"].split("_")[-1])

    return {
        "department": id2dept[dept_id],
        "department_confidence": round(dept_result["score"], 3),
        "urgency": id2urg[urg_id],
        "urgency_confidence": round(urg_result["score"], 3),
    }


def gradio_predict(text):
    result = predict_ticket(text)
    return (
        f"Department: {result['department']} (confidence: {result['department_confidence']})\n"
        f"Urgency: {result['urgency']} (confidence: {result['urgency_confidence']})"
    )


demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(lines=4, placeholder="Paste a customer support ticket here..."),
    outputs=gr.Textbox(label="Prediction"),
    title="SupportSense — Ticket Classifier & Router",
    description="Enter a customer support ticket to predict its department and urgency.",
    examples=[
        ["My laptop screen keeps flickering and won't stay on."],
        ["I was charged twice for the same order this month."],
        ["I can't log into my account and the reset email never arrives."],
    ],
)

if __name__ == "__main__":
    demo.launch()
