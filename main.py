import cv2
import pytesseract
import numpy as np
import spacy
import pandas as pd
import pdfplumber
from PIL import Image
import io
import re
import mysql.connector
from datetime import datetime
import tempfile
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Document Processor API",
             description="API for processing and extracting information from documents",
             version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only. In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Set Tesseract path - you may need to adjust this based on your system
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Caskroom/miniconda/base/bin/tesseract'
# For Linux users, might be: r'/usr/bin/tesseract'
# For Windows users, might be: r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# MySQL database connection details
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "ankit2061",
    "database": "ClaimDetails",
}

# Response models
class ExtractedField(BaseModel):
    Text: str
    Label: str

class ProcessingResponse(BaseModel):
    extracted_text: str
    extracted_info: List[ExtractedField]
    success: bool
    message: str

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
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
        print(f"Tesseract Error: {e}")
        return ""
    return text.strip()

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Check if text extraction was successful
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
    return text.strip()

def extract_aadhaar_number(text):
    """
    Specialized function to extract Aadhaar numbers using multiple approaches
    """
    # Clean the text first - this can help with OCR errors
    cleaned_text = re.sub(r'\s+', ' ', text)
    # Method 1: Look for common Aadhaar number patterns (12 digits with or without separators)
    aadhaar_patterns = [
        # Pattern with no separators - 12 consecutive digits
        r'(?<!\d)(\d{12})(?!\d)',
        # Pattern with spaces like "1234 5678 9012"
        r'(\d{4}\s+\d{4}\s+\d{4})',
        # Pattern with dashes like "1234-5678-9012"
        r'(\d{4}-\d{4}-\d{4})'
    ]
    
    for pattern in aadhaar_patterns:
        matches = re.search(pattern, cleaned_text)
        if matches:
            # Remove any spaces or dashes for consistent format
            aadhaar = re.sub(r'[-\s]', '', matches.group(0))
            return aadhaar
    
    return None

def process_text(text, key_phrases):
    """
    Process text to extract important information based on key phrases.
    Uses multiple approaches for robust extraction.
    """
    extracted_info = []
    found_labels = set()
    
    # 1. First pass: Look for key phrases and extract the following text
    for phrase in key_phrases:
        label = phrase.rstrip(':')
        pattern = fr'{re.escape(phrase)}\s*([^:]+?)(?=\s*(?:{"|".join([re.escape(p) for p in key_phrases])})|$)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        if matches:
            cleaned_text = matches[0].strip()
            if cleaned_text:
                extracted_info.append({"Text": cleaned_text, "Label": label})
                found_labels.add(label)
    
    # 2. Second pass: Use NLP to identify entities
    doc = nlp(text)
    
    for ent in doc.ents:
        label = None
        if ent.label_ == "PERSON" and "Name" not in found_labels:
            label = "Name"
        elif ent.label_ == "GPE" and "Address" not in found_labels:
            label = "Address"
        elif ent.label_ == "ORG" and "Hospital Name" not in found_labels:
            label = "Hospital Name"
        
        if label:
            cleaned_text = ent.text.strip()
            if cleaned_text:
                extracted_info.append({"Text": cleaned_text, "Label": label})
                found_labels.add(label)
    
    # Special handling for Aadhaar numbers
    if "Aadhar Card" not in found_labels:
        aadhaar = extract_aadhaar_number(text)
        if aadhaar:
            extracted_info.append({"Text": aadhaar, "Label": "Aadhar Card"})
            found_labels.add("Aadhar Card")
    
    # 3. Third pass: Look for unlabeled numbers that might be specific fields
    if "Phone Number" not in found_labels:
        # Look for potential phone numbers (10-digit sequences)
        phone_matches = re.search(r'(?<!\d)(\d{10})(?!\d)', text)
        if phone_matches:
            extracted_info.append({"Text": phone_matches.group(0), "Label": "Phone Number"})
            found_labels.add("Phone Number")
    
    if "Amount" not in found_labels:
        # Look for currency amounts
        amount_matches = re.search(r'(?:Rs\.?|INR)\s*(\d+(?:[,.]\d+)*)', text)
        if amount_matches:
            cleaned_text = re.sub(r'[^\d.]', '', amount_matches.group(0))
            if cleaned_text:
                extracted_info.append({"Text": cleaned_text, "Label": "Amount"})
                found_labels.add("Amount")
    
    return extracted_info

def save_to_database(data, insurance_id, file_path):
    try:
        cnx = mysql.connector.connect(**DB_CONFIG)
        cursor = cnx.cursor()
        
        # Insert document information into the 'patient_documents' table
        insert_doc_query = "INSERT INTO patient_documents (insurance_id, file_path) VALUES (%s, %s)"
        doc_values = (insurance_id, file_path)
        cursor.execute(insert_doc_query, doc_values)
        
        # Modified query to exclude the ventilation and appointment_time columns
        insert_patient_query = """
        INSERT INTO patient_details
        (insurance_id, name, father_name, aadhar_card, gender, blood_group,
        address, hospital_name, phone_number, amount,
        disease_name, disease_details, medicines, bed_type, other_charges)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        name = next((item["Text"] for item in data if item["Label"] == "Name"), None)
        father_name = next((item["Text"] for item in data if item["Label"] == "Father's Name"), None)
        aadhar_card = next((item["Text"] for item in data if item["Label"] == "Aadhar Card"), None)
        gender = next((item["Text"] for item in data if item["Label"] == "Gender"), None)
        blood_group = next((item["Text"] for item in data if item["Label"] == "Blood Group"), None)
        address = next((item["Text"] for item in data if item["Label"] == "Address"), None)
        hospital_name = next((item["Text"] for item in data if item["Label"] == "Hospital Name"), None)
        phone_number = next((item["Text"] for item in data if item["Label"] == "Phone Number"), None)
        
        # Clean the amount value
        amount = next((item["Text"] for item in data if item["Label"] == "Amount"), None)
        if amount:
            amount = re.sub(r'[^\d.]', '', amount)
            if amount:
                try:
                    amount = float(amount)
                except ValueError:
                    amount = None
        
        disease_name = next((item["Text"] for item in data if item["Label"] == "Disease Name"), None)
        disease_details = next((item["Text"] for item in data if item["Label"] == "Disease Details"), None)
        medicines = next((item["Text"] for item in data if item["Label"] == "Medicines"), None)
        bed_type = next((item["Text"] for item in data if item["Label"] == "Bed Type"), None)
        
        # Clean other_charges
        other_charges = next((item["Text"] for item in data if item["Label"] == "Other Charges"), None)
        if other_charges:
            other_charges = re.sub(r'[^\d.]', '', other_charges)
            if other_charges:
                try:
                    other_charges = float(other_charges)
                except ValueError:
                    other_charges = None
        
        # Note: ventilation and appointment_time are removed from the values tuple
        patient_values = (insurance_id, name, father_name, aadhar_card, gender, blood_group,
                          address, hospital_name, phone_number, amount,
                          disease_name, disease_details, medicines, bed_type, other_charges)
        
        cursor.execute(insert_patient_query, patient_values)
        cnx.commit()
        cursor.close()
        cnx.close()
        return True
    except mysql.connector.Error as err:
        print(f"Error saving to database: {err}")
        return False

async def process_file_api(file_path):
    try:
        text = ""
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            text = extract_text_from_image(file_path)
        elif file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        
        if not text.strip():
            return None, "No text detected in the file."
        
        important_info = process_text(text, ["Name:", "Father's Name:", "Aadhar Card:", "Gender:", "Blood Group:",
                                             "Address:", "Hospital Name:", "Insurance ID:", "Phone Number:",
                                             "Amount:", "Disease Name:", "Disease Details:", "Medicines:",
                                             "Bed Type:", "Ventilation:", "Other Charges:"])
        
        return text, important_info
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, str(e)

# API endpoints
@app.get("/")
async def root():
    return {"message": "Document Processing API is running"}

@app.post("/process/", response_model=ProcessingResponse)
async def process_document(file: UploadFile = File(...)):
    try:
        # Create a temporary file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            # Read the uploaded file and write to the temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the temporary file
        extracted_text, extracted_info = await process_file_api(temp_file_path)
        
        # Remove the temporary file
        os.unlink(temp_file_path)
        
        if extracted_text is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": extracted_info, "extracted_text": "", "extracted_info": []}
            )
        
        # Convert the extracted info to the expected format
        formatted_info = [{"Text": item["Text"], "Label": item["Label"]} for item in extracted_info]
        
        return ProcessingResponse(
            success=True,
            message="Document processed successfully",
            extracted_text=extracted_text,
            extracted_info=formatted_info
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error processing document: {str(e)}", "extracted_text": "", "extracted_info": []}
        )

@app.post("/save-to-database/")
async def save_document_data(file: UploadFile = File(...)):
    try:
        # Create a temporary file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            # Read the uploaded file and write to the temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the temporary file
        extracted_text, extracted_info = await process_file_api(temp_file_path)
        
        if extracted_text is None:
            # Remove the temporary file
            os.unlink(temp_file_path)
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": extracted_info}
            )
        
        # Get the Insurance ID
        insurance_id_data = next((item["Text"] for item in extracted_info if item["Label"] == "Insurance ID"), None)
        
        if insurance_id_data is None:
            # Remove the temporary file
            os.unlink(temp_file_path)
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Insurance ID not found. Data not saved."}
            )
        
        # Save the file to a permanent location (could be implemented based on requirements)
        permanent_file_path = f"uploads/{file.filename}"
        os.makedirs(os.path.dirname(permanent_file_path), exist_ok=True)
        with open(permanent_file_path, "wb") as perm_file:
            perm_file.write(content)
        
        # Remove the temporary file
        os.unlink(temp_file_path)
        
        # Save to database
        save_result = save_to_database(extracted_info, insurance_id_data, permanent_file_path)
        
        if save_result:
            return JSONResponse(
                content={"success": True, "message": "Data saved to database successfully."}
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": "Failed to save data to database."}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error saving to database: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)