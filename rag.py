from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings  # Or other embedding model
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

import nest_asyncio
 

nest_asyncio.apply()
loader = PyPDFLoader("Untitled document.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunked_documents = text_splitter.split_documents(pages)

chunked_documents[0]

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-l6-v2', encode_kwargs={'normalize_embeddings': False})

db = FAISS.from_documents(chunked_documents, embeddings)
#vectorstore = Chroma.from_documents(documents=chunked_documents, embedding=embeddings)
db.save("database")

question = "What is GovTech?"
searchDocs = db.similarity_search(question)
print(searchDocs[0].page_content)

#retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retriever=db.as_retriever()
docs = retriever.get_relevant_documents("What is Cheesemaking?")
print(docs[0].page_content)


from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser

llm = LlamaCpp(
    model_path="mistral7b_govtech_v2.gguf",
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)



from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is GovTech?")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
