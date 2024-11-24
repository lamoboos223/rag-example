import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores.chroma import Chroma


CHROMA_PATH = "chroma"

def main():
    print("Starting document processing...")
    # Combine argument parsing into a single parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--type", type=str, choices=['pdf', 'markdown'], default='pdf',
                       help="Type of documents to load (pdf or markdown)")
    args = parser.parse_args()
    
    if args.reset:
        print("✨ Clearing Database")
        clear_database()
        print("Database reset complete. Exiting...")
        return  # Add this line to exit after reset

    # Create (or update) the data store.
    print(f"Loading {args.type} documents...")
    documents = load_documents(args.type)  # Pass the type argument
    print(f"Found {len(documents)} documents")
    
    print("Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    print("Adding chunks to Chroma database...")
    add_to_chroma(chunks)
    print("Finished processing documents")


def load_documents(doc_type: str):
    if doc_type == "pdf":
        print("Loading PDF files from ./pdf directory...")
        document_loader = PyPDFDirectoryLoader("./pdf")
        try:
            documents = document_loader.load()
            print(f"Successfully loaded {len(documents)} PDF documents")
            return documents
        except Exception as e:
            print(f"Error loading PDF documents: {str(e)}")
            return []
    else:
        print("Loading Markdown files from ./markdown directory...")
        # Use a simpler, faster loading approach
        markdown_path = "./markdown"
        documents = []
        
        # Get list of markdown files
        import glob
        md_files = glob.glob(os.path.join(markdown_path, "**/*.md"), recursive=True)
        print(f"Found {len(md_files)} markdown files to process")
        
        # Process each file with basic markdown loading
        for file_path in md_files:
            print(f"Processing {file_path}...")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": file_path}
                    ))
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                
        print(f"Successfully loaded {len(documents)} documents")
        return documents


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    print("Initializing Chroma database...")
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    print("Calculating chunk IDs...")
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    print("Checking for existing documents...")
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        print("Adding documents to database...")
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    print("Starting chunk ID calculation...")
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} chunks...")

    print("Finished calculating chunk IDs")
    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()