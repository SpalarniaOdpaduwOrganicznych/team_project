import warnings
import re
warnings.filterwarnings('ignore')
import gradio as gr
from joblib import load
from texts import texts
from texts_2 import additional_info

# Imported models
clf_loaded = load("project_models/model_MNB_1.joblib")
vectorizer_loaded = load("project_models/tfidf_vectorizer_1.joblib")

def predict(text):
    transformed_text = vectorizer_loaded.transform([text])
    prediction = clf_loaded.predict(transformed_text)
    return prediction[0]

def get_additional_text():
    with open("project_tests/additional_text.txt", "r") as file:
        return file.read()

def find_special_text(text):
    for keyword, special_text in texts.items():
        if re.search(rf"\b{keyword}\b", text, re.IGNORECASE):
            return special_text
    return None

# Frontend part 
with gr.Blocks(css="""
    .gradio-container {
        background-color: #2e8b57;
    }
    h1 {
        color: #ffffff;
        font-size: 2em;
    }
    .description {
        font-size: 1.2em;
        color: #f5f5f5;
    }
    .explanation-box, .info-box {
        background-color: #4f4f4f;  # Grey background for both boxes
        padding: 10px;
        border: 2px solid #ffffff;
        border-radius: 5px;
        margin-top: 20px;
        color: #ffd700;
        display: none;  # Initially hidden
    }
    .explanation-box.visible, .info-box.visible {
        display: block;  # Only visible when triggered
    }
""") as demo:
    gr.Markdown("<h1>Vaccines Fake News Classification</h1>")
    gr.Markdown('<p class="description"></p>')
    
    text_input = gr.Textbox(label="Enter text for classification")
    output_label = gr.Label(label="Prediction")
    additional_text_md = gr.Markdown("", elem_classes=["explanation-box"])
    info_text_md = gr.Markdown("", elem_classes=["info-box"])

    def classify_and_display(text):
        special_text = find_special_text(text)
        prediction = predict(text)
        result = "Given text, with a high probability is TRUE, but we recommend further research" if prediction == 1 else "Given text, with a high probability is FALSE, but we recommend further research"
        explanation = ""
        elem_classes = ["explanation-box"]

        if prediction == 0:
            sign_off = "<p style='font-size:1.5em;'>Here is your explanation.</p>"
            if special_text:
                explanation = sign_off + special_text
            else:
                additional_text = get_additional_text()
                explanation = sign_off + additional_text
            elem_classes.append("visible")

        return result, gr.update(value=explanation, elem_classes=elem_classes)

    more_info_button = gr.Button("More Information")
    more_info_button.click(
        fn=lambda: gr.update(value=additional_info, elem_classes=["info-box", "visible"]),
        inputs=[],
        outputs=info_text_md
    )

    classify_button = gr.Button("Classify")
    classify_button.click(fn=classify_and_display, inputs=text_input, outputs=[output_label, additional_text_md])

demo.launch()
