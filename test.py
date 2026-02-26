s = "ollo"


def pallindrome(s):
    c=0
    n=len(s)
    for i in range(n//2):
        if s[i] == s[n-i-1] :
            c+=1
            continue
        else:
            return "not a pallindrome"
    if c==n:
        return "pallindrome"
    return "not a pallindrome"
s ="a man a plan a canal panama"
s = s.split(" ")
s = "".join(s)
print(s)
print(pallindrome(s))

# Quick RAG setup in 6 lines
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Load and split
loader = PyPDFLoader("data.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Size of each chunk
    chunk_overlap=200       # Overlap between chunks for context
)
chunks = text_splitter.split_documents(documents)

# 2. Embed and store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Return top 3 most relevant chunks
)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0  # 0 = deterministic, 1 = creative
)

prompt_template = """
Use the following context to answer the question.
If you don't know the answer, say so. Don't make up information.
Context: {context}
Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" = put all context in one prompt
    retriever=retriever,
    return_source_documents=True,  # Return source chunks used
    chain_type_kwargs={"prompt": PROMPT}
)

# ============================================
# STEP 9: QUERY THE SYSTEM
# ============================================
# Ask questions and get answers
query = "What is the main topic of the document?"
result = qa_chain({"query": query})

print("Answer:", result["result"])
print("Source Documents:", result["source_documents"])
