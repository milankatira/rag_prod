from dotenv import load_dotenv
load_dotenv()
import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (TextLoader)

def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Hello, world! This is a test file.")
        temp_file_path = temp_file.name

    try:
        #load the text file using the TextLoader
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        # Print the loaded documents
        for document in documents:
            print(document)
            print(document.page_content)

    except Exception as e:
        print(f"Error loading text file: {e}")
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

if __name__ == "__main__":
    load_text_file()