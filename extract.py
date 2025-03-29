import cv2
import pytesseract
import numpy as np
import spacy
import pandas as pd
import pdfplumber
from PIL import Image
from IPython.display import display, Markdown
import ipywidgets as widgets
from tkinter import Tk, filedialog
import os

nlp = spacy.load("en_core_web_sm")

# Output widget for displaying extracted text and information
output_widget = widgets.Output()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        with output_widget:
            print("Error: Could not read the image. Please check the file format and path.")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    kernel = np.ones((1, 1), np.uint8)
    processed_image = cv2.dilate(binary, kernel, iterations=1)
    processed_image = cv2.erode(processed_image, kernel, iterations=1)

    return processed_image

def extract_text_from_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is None:
        return ""

    pil_image = Image.fromarray(preprocessed_image)
    try:
        text = pytesseract.image_to_string(pil_image)
    except pytesseract.TesseractError as e:
        with output_widget:
            output_widget.clear_output()
            print(f"Tesseract Error: {e}")
        return ""
    return text.strip()

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        with output_widget:
            output_widget.clear_output()
            print(f"Error reading PDF: {e}")
        return ""
    return text.strip()

def process_text(text, keywords=[]):
    doc = nlp(text)
    extracted_info = []

    for ent in doc.ents:
        if any(keyword.lower() in ent.text.lower() for keyword in keywords):
            extracted_info.append({"Text": ent.text, "Label": ent.label_})

    return extracted_info

def select_file(_):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select a File", filetypes=[
        ("Image Files", "*.jpg;*.jpeg;*.png"),
        ("PDF Files", "*.pdf")
    ])

    if file_path and os.path.exists(file_path):
        process_file(file_path)
    else:
        with output_widget:
            output_widget.clear_output()
            print("Error: No valid file selected.")

def process_file(file_path):
    try:
        text = ""
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            text = extract_text_from_image(file_path)
        elif file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        
        with output_widget:
            output_widget.clear_output()
            if not text.strip():
                print("Error: No text detected in the file.")
                return

            print("Full Extracted Text:", text)

            important_info = process_text(text, ["date", "amount", "name", "address", "Name:", "Customer Name", "Recipient", "Amount:", "Date:", "Address:", "ID:"])
            if important_info:
                display(Markdown("### Important Extracted Information:"))
                df = pd.DataFrame(important_info)
                display(df)
            else:
                display(Markdown("**No important information found.**"))
    except Exception as e:
        with output_widget:
            output_widget.clear_output()
            print(f"An error occurred: {e}")

button = widgets.Button(description="Select File")
button.on_click(select_file)

display(button, output_widget)

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Caskroom/miniconda/base/bin/tesseract'
