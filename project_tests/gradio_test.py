import warnings
import re
warnings.filterwarnings('ignore')
import gradio as gr
from joblib import load
from texts import texts

# imported models
clf_loaded = load("project_models/model_MNB_1.joblib")
vectorizer_loaded = load("project_models/tfidf_vectorizer_1.joblib")

def predict(text):
    transformed_text = vectorizer_loaded.transform([text])
    prediction = clf_loaded.predict(transformed_text)
    return prediction[0]

def get_additional_text():
    with open("project_tests/pies.txt", "r") as file:
        return file.read()

def find_special_text(text):
    for keyword, special_text in texts.items():
        if re.search(rf"\b{keyword}\b", text, re.IGNORECASE):
            return special_text
    return None

# """"frontend from hell"""""
with gr.Blocks(css="""
    .gradio-container {background-color: #348210;}
    h1 {color: black; font-size: 2em;}
    .description {font-size: 1.2em;}
    .explanation-box {
        background-color: grey;
        padding: 10px;
        border: 2px solid black;
        border-radius: 5px;
        margin-top: 20px;
        color: yellow;
        display: none;
    }
    .explanation-box.visible {
        display: block;
    }
""") as demo:
    gr.Markdown("<h1>Vaccines Fake News Classification</h1>")
    gr.Markdown('<p class="description">Strawberry fields forever</p>')
    
    text_input = gr.Textbox(label="Enter text for classification")
    output_label = gr.Label(label="Prediction")
    additional_text_md = gr.Markdown("", elem_classes=["explanation-box"])

    def classify_and_display(text):
        special_text = find_special_text(text)
        prediction = predict(text)
        result = "Given text, with a high probability is TRUE" if prediction == 1 else "Given text, with a high probability is FALSE"
        sign_off = "<p style='font-size:1.5em;'>Here is your explanation.</p>"
        if special_text:
            explanation = sign_off + special_text
        else:
            additional_text = get_additional_text()
            explanation = sign_off + additional_text

        return result, gr.update(value=explanation, elem_classes=["explanation-box", "visible"])

    btn = gr.Button("Classify")
    btn.click(fn=classify_and_display, inputs=text_input, outputs=[output_label, additional_text_md])

demo.launch()
