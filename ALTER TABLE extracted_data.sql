ALTER TABLE extracted_data
ADD COLUMN document_id INT,
ADD CONSTRAINT fk_document_id
FOREIGN KEY (document_id)
REFERENCES documents(id)
ON DELETE CASCADE;