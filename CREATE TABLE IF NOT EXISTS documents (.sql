CREATE TABLE IF NOT EXISTS documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_path VARCHAR(255) NOT NULL,
    processing_date DATETIME DEFAULT CURRENT_TIMESTAMP
    -- You can add other relevant information about the document here
);