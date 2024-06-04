from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import QAGenerationChain
from langchain_huggingface import HuggingFacePipeline

text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=50, chunk_size=1000)
loader = TextLoader("./files/ejemplo.txt", encoding="utf-8")
doc = loader.load()
texts = text_splitter.split_documents(doc)

llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Meta-Llama-3-8B",
    device_map="auto",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "top_k": 50,
        "temperature": 0.1,
        "do_sample": True,
    },
    
)

chain = QAGenerationChain.from_llm(llm=llm, text_splitter=text_splitter)
qa = chain.invoke(texts[0].page_content)
print(qa)