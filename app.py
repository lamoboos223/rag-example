import torch
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import faiss


print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name(0)}")

# Step 1: Creating the pdf file ✓
# Step 2: Loading the pdf file using the LangChain PDF loader ✓
loader = PyPDFLoader("./ThaiRecipes.pdf")
docs = loader.load()
print(f"Length of documents before chunking: {len(docs)}")

# Step 3: Create Chunks from the previously created documents ✓
splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=30)
chunked_docs = splitter.split_documents(docs)
print(f"Length of documents after chunking: {len(chunked_docs)}")

# Step 4: Create Embeddings for the chunked documents
# 4.1. Import the model BAAI/bge-base-en-v1.5 from HuggingFace And create the vector DB using FAISS library ✓
faiss.omp_set_num_threads(4)  # Optional: control CPU threads
db = FAISS.from_documents(chunked_docs, HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={'device': 'cpu'}  # Explicitly set to CPU
))
# 4.2. Derieve the Retriever out of the vector DB ✓
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Save the FAISS index to disk
# db.save_local("faiss_index")

# Later, you can load it back using:
# db = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"))

# Step 5: Load quantized model (The LLM that will be used for text generation) ✓
model_name = "mistralai/Ministral-8B-Instruct-2410"
# model_name = "distilbert/distilgpt2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 6: Setup the LLM chain 
# 6.1. Create a text_generation pipeline using the loaded model and its tokenizer ✓
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=700,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
# 6.2. Create a prompt template ✓
prompt_template = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
 """

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)
llm_chain = prompt | llm | StrOutputParser()
# 6.3. Combine the llm_chain with the retriever to create a RAG chain ✓
retriever = db.as_retriever()
print(f"Retriever: {retriever}")
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

# Test ✓
question = "What are the ingredients for Som Tum?"
without_rag_result = llm_chain.invoke({"context": "", "question": question})
with_rag_result = rag_chain.invoke(question)
print(f"Without RAG:\t {without_rag_result}\nWith RAG:\t {with_rag_result}")

