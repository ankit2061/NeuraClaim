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
from sqlalchemy import create_engine, Column, String, Integer, MetaData, Table, insert, select
from sqlalchemy.orm import sessionmaker

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

# SQLAlchemy Database Configuration (Replace with your actual database URL)
DATABASE_URL = "mysql+mysqlconnector://{user}:{password}@{host}/{database}".format(
    user=DB_CONFIG["user"],
    password=DB_CONFIG["password"],
    host=DB_CONFIG["host"],
    database=DB_CONFIG["database"]
)
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Define Table schemas
patient_details = Table(
    "patient_details",
    metadata,
    Column("insurance_id", String, primary_key=True),
    Column("name", String),
    Column("father_name", String),
    Column("aadhar_card", String),
    Column("gender", String),
    Column("blood_group", String),
    Column("address", String),
    Column("hospital_name", String),
    Column("phone_number", String),
    Column("amount", String),
    Column("claim_type", String),
    Column("disease_details", String),
    Column("medicines", String),
    Column("bed_type", String),
    Column("other_charges", String),
)

patient_documents = Table(
    "patient_documents",
    metadata,
    Column("insurance_id", String, primary_key=True),
    Column("file_path", String),
)

metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Response models
class ExtractedField(BaseModel):
    Text: str
    Label: str

class ProcessingResponse(BaseModel):
    extracted_text: str
    extracted_info: List[ExtractedField]
    success: bool
    message: str

# Define FormData class before using it
class FormData(BaseModel):
    insurance_id: Optional[str] = None
    name: Optional[str] = None
    father_name: Optional[str] = None
    aadhar_card: Optional[str] = None
    gender: Optional[str] = None
    blood_group: Optional[str] = None
    address: Optional[str] = None
    hospital_name: Optional[str] = None
    phone_number: Optional[str] = None
    amount: Optional[str] = None
    claim_type: Optional[str] = None
    disease_details: Optional[str] = None
    medicines: Optional[str] = None
    bed_type: Optional[str] = None
    other_charges: Optional[str] = None

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
    # This catches raw numbers that look like Aadhaar numbers
    aadhaar_patterns = [
        # Pattern with no separators - 12 consecutive digits
        r'(?<!\d)(\d{12})(?!\d)',
        # Pattern with space separators
        r'(\d{4}\s+\d{4}\s+\d{4})',
        # Pattern with dash separators
        r'(\d{4}-\d{4}-\d{4})',
        # Pattern with dot separators
        r'(\d{4}\.\d{4}\.\d{4})'
    ]
    
    for pattern in aadhaar_patterns:
        matches = re.findall(pattern, cleaned_text)
        if matches:
            # Clean up the found number (remove spaces, dashes)
            aadhaar = re.sub(r'[^\d]', '', matches[0])
            if len(aadhaar) == 12:
                return aadhaar
    
    # Method 2: Look for Aadhaar numbers with keywords
    # This catches Aadhaar numbers that are labeled
    keyword_patterns = [
        # Various ways "Aadhaar" might be written followed by a number
        r'(?:aadhar|aadhaar|adhar|aadha+r|आधार)(?:\s*(?:card|number|no|id|#|:|नंबर|संख्या))?\s*[:\.\-]?\s*((?:\d[\d\s\.\-]*){12})',
        r'(?:uid|unique\s+id)(?:\s*(?:number|no|#))?\s*[:\.\-]?\s*((?:\d[\d\s\.\-]*){12})',
        # Looking for "No:" or "Number:" followed by what could be an Aadhaar
        r'(?:no|number|id)?\s*[:\.\-]\s*((?:\d[\d\s\.\-]*){12})'
    ]
    
    for pattern in keyword_patterns:
        matches = re.findall(pattern, cleaned_text.lower())
        if matches:
            # Clean up the found number
            aadhaar = re.sub(r'[^\d]', '', matches[0])
            if len(aadhaar) == 12:
                return aadhaar
    
    # Method 3: More aggressive - find any 12-digit sequence that could be an Aadhaar number
    # Use with caution as it might pick up other 12-digit numbers
    digit_sequences = re.findall(r'(?<!\d)(\d[\d\s\.\-]*\d)(?!\d)', cleaned_text)
    for seq in digit_sequences:
        digits_only = re.sub(r'[^\d]', '', seq)
        if len(digits_only) == 12:
            return digits_only
            
    return None

def clean_extracted_field(text, field_type):
    """
    Cleans extracted text based on field type to remove common OCR artifacts
    and mislabeled content
    """
    # Convert to string in case we received another type
    text = str(text).strip()
    
    # Remove common label text that might be captured within the value
    unwanted_labels = [
        "Phone Number", "Contact", "Mobile", "Call",
        "Hospital Name", "Doctor", "Clinic", "MD", "Dr\\.",
        "Address", "Location", "Place", "Residence",
        "Insurance ID", "Policy Number", "Insurance",
        "Amount", "Total", "Fee", "Payment",
        "Disease", "Diagnosis", "Condition",
        "Medicines", "Medication", "Drugs", "Prescription"
    ]
    
    # For each unwanted label, try to remove it if it appears at the end
    for label in unwanted_labels:
        # Create pattern to match label at the end of the text (allowing for spaces)
        pattern = rf'\s*{re.escape(label)}$'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove common field separators
    text = re.sub(r'[:;|]$', '', text)
    
    # Clean up newlines and extra spaces
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Additional field-specific cleaning
    if field_type in ["Address"]:
        # Keep only relevant address information
        text = re.sub(r'\s*(?:Phone|Mobile|Contact|Email).*$', '', text, flags=re.IGNORECASE)
    
    elif field_type in ["Hospital Name"]:
        # Remove doctor references
        text = re.sub(r'\s*(?:Doctor|Dr\.|MD|Physician).*$', '', text, flags=re.IGNORECASE)
    
    elif field_type in ["Phone Number"]:
        # Keep only digits and basic formatting characters
        text = re.sub(r'[^\d+\-\s()]', '', text)
    
    return text.strip()

def extract_fields_with_boundaries(text):
    """
    Extract fields with improved boundary detection to prevent label bleed
    """
    extracted_info = []
    found_labels = set()
    
    # Dictionary of field patterns with better boundary detection
    field_patterns = {
        "Name": r'(?:Patient(?:\s*Name)?|Name|Patient)[:;]?\s*([\w\s\.]+?)(?=\n|$|(?:Father|Gender|Blood|Aadhaar))',
        "Father's Name": r'(?:Father(?:[\'s]*\s*Name)?|Father)[:;]?\s*([\w\s\.]+?)(?=\n|$|(?:Gender|Blood|Aadhaar))',
        "Gender": r'(?:Gender|Sex)[:;]?\s*(Male|Female|Other|M|F)(?=\n|$)',
        "Blood Group": r'(?:Blood(?:\s*Group)?)[:;]?\s*([ABO][+-]|AB[+-])(?=\n|$)',
        "Address": r'(?:Address|Location|Place|Residence)[:;]?\s*([\w\s,\.\-\/]+?)(?=\n|$|(?:Phone|Mobile|Contact|Email))',
        "Hospital Name": r'(?:Hospital(?:\s*Name)?|Clinic|Medical Center)[:;]?\s*([\w\s\.]+?)(?=\n|$|(?:Doctor|Dr|MD|Address))',
        "Insurance ID": r'(?:Insurance(?:\s*(?:ID|Number|No))?|Policy(?:\s*Number)?)[:;]?\s*([\w\d\-]+?)(?=\n|$)',
        "Phone Number": r'(?:Phone(?:\s*Number)?|Mobile|Contact|Cell)[:;]?\s*([\d\s\+\-\(\)]+?)(?=\n|$)',
        "Amount": r'(?:Amount|Total|Cost|Fee|Charges)[:;]?\s*([\d\.]+?)(?=\n|$|Rs|\$|₹)',
        "Disease Name": r'(?:Disease(?:\s*Name)?|Diagnosis|Condition|Ailment)[:;]?\s*([\w\s]+?)(?=\n|$|(?:Disease Details|Symptoms|Treatment))',
        "Disease Details": r'(?:Disease(?:\s*Details)?|Details|Diagnosis Details|Clinical Details|Symptoms)[:;]?\s*([\w\s,\.;\(\)\-\/]+?)(?=\n\n|\n(?:Medicines|Medications|Drugs)|$)',
        "Medicines": r'(?:Medicines|Medications|Drugs|Prescriptions|Medicine List)[:;]?\s*([\w\s,\.;\(\)\-\/]+?)(?=\n\n|\n(?:Bed|Ventilation|Amount|Charges)|$)',
        "Bed Type": r'(?:Bed(?:\s*Type)?)[:;]?\s*([\w\s]+?)(?=\n|$)',
        "Ventilation": r'(?:Ventilation|Ventilator|Oxygen)[:;]?\s*(Yes|No|Required|Not Required)(?=\n|$)',
        "Other Charges": r'(?:Other(?:\s*Charges)?|Additional(?:\s*Charges)?|Extra)[:;]?\s*([\d\.]+?)(?=\n|$|Rs|\$|₹)'
    }
    
    # 1. First pass: Extract Aadhaar number with dedicated function
    aadhaar = extract_aadhaar_number(text)
    if aadhaar:
        formatted_aadhaar = f"{aadhaar[:4]}-{aadhaar[4:8]}-{aadhaar[8:]}"
        extracted_info.append({"Text": formatted_aadhaar, "Label": "Aadhar Card"})
        found_labels.add("Aadhar Card")
    
    # 2. Second pass: Extract other fields with improved boundary detection
    for label, pattern in field_patterns.items():
        if label in found_labels:
            continue
            
        matches = re.search(pattern, text, re.IGNORECASE)
        if matches:
            extracted_text = matches.group(1).strip()
            # Clean the extracted text to remove potential label contamination
            cleaned_text = clean_extracted_field(extracted_text, label)
            
            # Only add if we have meaningful content
            if cleaned_text and len(cleaned_text) > 0:
                extracted_info.append({"Text": cleaned_text, "Label": label})
                found_labels.add(label)
    
    # 3. Third pass: Look for unlabeled numbers that might be specific fields
    if "Phone Number" not in found_labels:
        # Look for potential phone numbers (10-digit sequences)
        phone_matches = re.search(r'(?<!\d)(\d{10})(?!\d)', text)
        if phone_matches:
            extracted_info.append({"Text": phone_matches.group(1), "Label": "Phone Number"})
            found_labels.add("Phone Number")
    
    # Look for Appendicitis or other common conditions if disease name not found
    if "Disease Name" not in found_labels:
        common_diseases = ["appendicitis", "diabetes", "hypertension", "cancer", "fracture", "pneumonia"]
        for disease in common_diseases:
            if re.search(rf'\b{disease}\b', text, re.IGNORECASE):
                extracted_info.append({"Text": disease.capitalize(), "Label": "Disease Name"})
                found_labels.add("Disease Name")
                break
    
    return extracted_info

def process_text(text, keywords=[]):
    """
    Main processing function that combines extraction methods
    """
    # Get fields using improved boundary detection
    extracted_info = extract_fields_with_boundaries(text)
    
    # For backward compatibility, still use keyword-based extraction for any missing fields
    found_labels = {item["Label"] for item in extracted_info}
    
    for keyword in keywords:
        # Skip keywords for fields we already found
        label = keyword.replace(":", "").strip()
        if any(label in existing for existing in found_labels):
            continue
            
        # Simple keyword-based extraction as fallback
        pattern = re.compile(rf"{re.escape(keyword)}\s*([\w\s\d\.\-]+?)(?=\n|$)", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            extracted_text = match.group(1).strip()
            cleaned_text = clean_extracted_field(extracted_text, label)
            
            if cleaned_text and len(cleaned_text) > 0:
                extracted_info.append({"Text": cleaned_text, "Label": label})
                found_labels.add(label)
    
    # Additionally, use NLP for entity recognition as a fallback for missing fields
    if "Name" not in found_labels or "Hospital Name" not in found_labels or "Address" not in found_labels:
        doc = nlp(text)
        
        for ent in doc.ents:
            if ent.label_ == "PERSON" and "Name" not in found_labels:
                extracted_info.append({"Text": ent.text.strip(), "Label": "Name"})
                found_labels.add("Name")
            elif ent.label_ == "GPE" and "Address" not in found_labels:
                extracted_info.append({"Text": ent.text.strip(), "Label": "Address"})
                found_labels.add("Address")
            elif ent.label_ == "ORG" and "Hospital Name" not in found_labels:
                extracted_info.append({"Text": ent.text.strip(), "Label": "Hospital Name"})
                found_labels.add("Hospital Name")
    
    return extracted_info

def save_to_database(extracted_info: List[Dict[str, str]], insurance_id: str, permanent_file_path: str) -> bool:
    """Saves extracted data to the database."""
    db = SessionLocal()
    try:
        patient_data = {
            "insurance_id": insurance_id,
            "name": next((item["Text"] for item in extracted_info if item["Label"] == "Name"), None),
            "father_name": next((item["Text"] for item in extracted_info if item["Label"] == "Father's Name"), None),
            "aadhar_card": next((item["Text"] for item in extracted_info if item["Label"] == "Aadhar Card"), None),
            "gender": next((item["Text"] for item in extracted_info if item["Label"] == "Gender"), None),
            "blood_group": next((item["Text"] for item in extracted_info if item["Label"] == "Blood Group"), None),
            "address": next((item["Text"] for item in extracted_info if item["Label"] == "Address"), None),
            "hospital_name": next((item["Text"] for item in extracted_info if item["Label"] == "Hospital Name"), None),
            "phone_number": next((item["Text"] for item in extracted_info if item["Label"] == "Phone Number"), None),
            "amount": next((item["Text"] for item in extracted_info if item["Label"] == "Amount"), None),
            "claim_type": next((item["Text"] for item in extracted_info if item["Label"] == "Claim Type"), None),
            "disease_details": next((item["Text"] for item in extracted_info if item["Label"] == "Disease Details"), None),
            "medicines": next((item["Text"] for item in extracted_info if item["Label"] == "Medicines"), None),
            "bed_type": next((item["Text"] for item in extracted_info if item["Label"] == "Bed Type"), None),
            "other_charges": next((item["Text"] for item in extracted_info if item["Label"] == "Other Charges"), None),
        }

        db.execute(insert(patient_details).values(patient_data))

        document_data = {
            "insurance_id": insurance_id,
            "file_path": permanent_file_path,
        }
        db.execute(insert(patient_documents).values(document_data))

        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Database error: {e}") #Print the error for debugging.
        return False
    finally:
        db.close()

async def process_file_api(file_path):
    try:
        text = ""
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            text = extract_text_from_image(file_path)
        elif file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        
        if not text.strip():
            return None, "No text detected in the file."
        
        key_phrases = ["Name:", "Father's Name:", "Aadhar Card:", "Gender:", "Blood Group:",
                      "Address:", "Hospital Name:", "Insurance ID:", "Phone Number:",
                      "Amount:", "Disease Details:", "Medicines:",
                      "Bed Type:", "Other Charges:"]
                      
        important_info = process_text(text, key_phrases)
        
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

@app.get("/get-latest-form-data")
async def get_latest_form_data():
    try:
        cnx = mysql.connector.connect(**DB_CONFIG)
        cursor = cnx.cursor()
        query = """
            SELECT insurance_id, name, father_name, aadhar_card, gender, blood_group,
                           address, hospital_name, phone_number, amount, disease_name, disease_details
                FROM patient_details
                ORDER BY insurance_id DESC
                LIMIT 1
            """
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()
        cnx.close()

        print(result)
        if result:
            columns = ["insurance_id", "name", "father_name", "aadhar_card", "gender", "blood_group",
                           "address", "hospital_name", "phone_number", "amount", "disease_name", "disease_details"]
            data = dict(zip(columns, result))
            return {"success": True, "data": data}
        else:
            return {"success": False, "message": "No data found."}

    except mysql.connector.Error as err:
        print(err)
        return {"success": False, "message": f"Database error: {err}"}

@app.get("/get-patient-details/{insurance_id}")
async def get_patient_details(insurance_id: str):
    try:
        db = SessionLocal()
        # Use the exact columns from your table definition
        query = select([
            patient_details.c.insurance_id,
            patient_details.c.name,
            patient_details.c.father_name,
            patient_details.c.aadhar_card,
            patient_details.c.gender,
            patient_details.c.blood_group,
            patient_details.c.address,
            patient_details.c.hospital_name,
            patient_details.c.phone_number,
            patient_details.c.amount,
            patient_details.c.claim_type,
            patient_details.c.disease_details,
            patient_details.c.medicines,
            patient_details.c.bed_type,
            patient_details.c.other_charges
        ]).where(patient_details.c.insurance_id == insurance_id)
        
        result = db.execute(query).fetchone()
        db.close()
        
        if result:
            # Convert SQLAlchemy row to dictionary with proper handling for Decimal values
            data = {}
            for key, value in dict(result).items():
                if isinstance(value, decimal.Decimal):
                    data[key] = float(value)
                else:
                    data[key] = value
                    
            return JSONResponse(content={"success": True, "data": data})
        else:
            return JSONResponse(content={"success": False, "message": f"No data found for insurance ID: {insurance_id}"})

    except Exception as e:
        return JSONResponse(content={"success": False, "message": f"Database error: {str(e)}"})
@app.post("/submit-verified-data/")
async def submit_verified_data(form_data: FormData):
    try:
        cnx = mysql.connector.connect(**DB_CONFIG)
        cursor = cnx.cursor()
        query = """
            CREATE patient_details
            SET name = %s, father_name = %s, aadhar_card = %s, gender = %s, blood_group = %s,
                address = %s, hospital_name = %s, doctor_name = %s, appointment_time = %s,
                phone_number = %s, amount = %s, claim_type = %s
            WHERE insurance_id = %s
        """
        values = (form_data.name, form_data.fathers_name, form_data.aadhaar, form_data.gender, form_data.blood_group,
                  form_data.address, form_data.hospital_name, form_data.doctor_name, form_data.appointment_time,
                  form_data.phone, form_data.amount, form_data.claim_type, form_data.insurance_id)
        print(values)
        cursor.execute(query, values)
        cnx.commit()
        cursor.close()
        cnx.close()
        return {"success": True, "message": "Data updated successfully."}
    except mysql.connector.Error as err:
        return {"success": False, "message": f"Database error: {err}"}

if __name__ == "__main__":
    print([{"path": route.path, "name": route.name} for route in app.routes])

    uvicorn.run(app, host="0.0.0.0", port=8000)