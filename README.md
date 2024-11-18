## <center>Information Retrieval - Final Project</center>
### <center> —— Implementing a Football Domain-Specific Knowledge Dialogue Model with Enhanced Retrieval Using LlamaIndex Framework </center>

### 1. Project Requirements

Large language models (LLMs), such as ChatGPT, have demonstrated powerful capabilities in natural language understanding and generation. Retrieval-Augmented Generation (RAG) is a technique that retrieves relevant information from external knowledge sources before generating responses using LLMs. This allows LLMs to generate more accurate and contextually relevant answers by leveraging external knowledge resources.

This project requires constructing a system capable of answering 2024 football news questions using the `LlamaIndex` framework, combining large language models with RAG techniques.

![alt text](picture/image.png)

Specific requirements include:

> - Dataset: Use the provided dataset of 2024 football news articles (`data.csv`).
> - Basic RAG System Implementation: Use the `LlamaIndex` framework to construct a basic RAG system, including indexing, retrieval, and generation steps.
> - Improvements and Optimization: Experiment with different chunking strategies and sizes, embedding models, multi-level indexing, query rewriting, hybrid retrieval, and re-ranking to optimize system performance.
> - Evaluation: Design experiments to evaluate the basic and improved systems, providing performance metrics and examples.

### 2. Implementation Approach

Based on the project requirements, the implementation approach is outlined as follows:

![alt text](<picture/未命名文件 (2).png>)

First, a simple dialogue model is built using `LlamaIndex`. Optimization is then performed in three areas: indexing methods, hybrid retrieval, and query rewriting. Finally, the responses from the basic model and the optimized model are evaluated using four metrics within the `DeepEval` framework to compare their performance.

### 3. Basic Dialogue Model with `LlamaIndex`

#### 1. OpenAI API

```python
import os
from dotenv import load_dotenv

OPENAI_API_KEY = 'OPENAI_API_KEY' 
os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY' 

load_dotenv()
```

#### 2. Dataset Preparation

Load the `documents` by reading the files using the `SimpleDirectoryReader` method and saving them as `documents`.

```python
import os
import gradio as gr
import openai

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,
ServiceContext,PromptTemplate
from llama_index.core.schema import IndexNode

from llama_index.core import (
    GPTKeywordTableIndex,
    SimpleDirectoryReader,
    ServiceContext
)

documents = SimpleDirectoryReader(input_dir='./data').load_data()
```

#### 3. Create Index Using `OpenAIEmbeddings`

Use the most basic indexing method, `OpenAIEmbeddings`. This process will automatically create `nodes` from the `documents`.

```python
from llama_index.core import VectorStoreIndex,DocumentSummaryIndex
from langchain_openai import OpenAIEmbeddings

# OpenAIEmbeddings()  
print("OpenAIEmbeddings:")
index_OpenAIEmbeddings = VectorStoreIndex.from_documents(documents = documents,
 embedding = OpenAIEmbeddings(), show_progress = 1)
```

Indexing process:

![alt text](picture/image-1.png)

#### 4. Generate Responses with the RAG Retrieval Engine `get_response(query)`

```python
# Different embedding-based engines can be used in this section
query_engine = index_OpenAIEmbeddings.as_chat_engine(verbose=True)

# Define the get_response function
def get_response(query):
    response = query_engine.query(query)
    return response
```

#### 5. `Gradio` Interface

```python
import gradio as gr
from gradio.components import Textbox

# Create the Gradio interface
iface = gr.Interface(
    fn=get_response,
    inputs=gr.components.Textbox(lines=5, label="Input your query"),
    outputs="text",
    title="Luxinai's Football Information Retrieval System",
    description="RAG-based system using GPT-3.5 + the latest football news"
)

# Launch the Gradio interface
iface.launch()
```

【Interface and Q&A Effect】Input a question, click the `Submit` button to submit the query, and the engine will display the response after enhanced retrieval in the `output` dialogue box on the right. Users can click `Flagged` to save both the query and the engine's response to a backend file. Users can also click the `Clear` button on the left to reset the input and ask a new question.

![alt text](picture/8abea2cac30f25fef7fb7a50b756e3e.png)
```
### Four: Model Optimization  
#### 1. Two Different `HuggingFace` Embedding Models  
##### (1) `bge-large-zh-v1.5`  

![alt text](image-2.png)  

![alt text](image-3.png)  

Define `index_bge_large_zh` as the index name using this embedding model.  

```python  
# bge-large-zh-v1.5  
print("bge-large-zh-v1.5:")  
bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")  
index_bge_large_zh = VectorStoreIndex.from_documents(documents = documents,  
                embedding = bge_embeddings, show_progress = 1)  
```  
Example `response` generation:  

##### (2) `bge-M3`  

`BGE-M3` is the first semantic vector model that integrates three technical features: Multi-Linguality, Multi-Granularity, and Multi-Functionality, greatly improving the usability of semantic vector models in real-world applications. Currently, `BGE-M3` is fully open-sourced to the community.  

Define `index_bge_M3` as the index name using this embedding model.  

```python  
# bge-M3  
print("bge-M3:")  
bgeM3_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-M3")  
index_bge_M3 = VectorStoreIndex.from_documents(documents = documents,  
                embedding = bgeM3_embeddings, show_progress = 1)  
```  
Example `response` generation: It can be seen that when asked "What kind of model are you?", the enhanced engine adds football-related private domain knowledge.  

![alt text](picture/image-4.png)  

#### 2. Hybrid Retrieval  
##### (1) Preprocess the Document Set  
Convert `documents` into a format suitable for `.from_texts()` initialization method of `BM25Retriever` and `FAISS`.  

```python  
# Define an empty text list  
doc_texts = []  
splitted_texts = []  

# Iterate through each document object to retrieve text content and metadata  
for i, doc in enumerate(documents):  
    text = doc.text  # Use the text attribute to get the content  
    if text:  
        doc_texts.append(text)  
    else:  
        doc_texts.append("")  # Add an empty string if the content is empty  

# Split each text content by "http"  
for text in doc_texts:  
    splitted_texts.extend(text.split("http"))  

# Convert documents into a list of texts and metadata  
doc_metadatas = [{"source": i} for i in range(len(splitted_texts))]  
```  

##### (2) Define Sparse Retrieval `BM25Retriever` and Dense Retrieval `FAISS`  

```python  
# Attempt hybrid retrieval  
from langchain_community.retrievers import BM25Retriever  
from langchain_community.vectorstores import FAISS  
from langchain_openai import OpenAIEmbeddings  

# Initialize BM25 retriever  
bm25_retriever = BM25Retriever.from_texts(splitted_texts, metadatas = doc_metadatas)  
bm25_retriever.k = 3  

# Initialize FAISS retriever  
embedding = OpenAIEmbeddings()  
faiss_vectorstore = FAISS.from_texts(splitted_texts, embedding, metadatas=doc_metadatas)  

# Convert FAISS vector store into a retriever  
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})  
```  

##### (3) Define Hybrid Retrieval `EnsembleRetriever`  

```python  
# Attempt hybrid retrieval  
from langchain.retrievers import EnsembleRetriever  
from langchain_openai import OpenAIEmbeddings  

# Initialize Ensemble Retriever  
ensemble_retriever = EnsembleRetriever(  
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]  
)  
```  

##### (4) Use Hybrid Retrieval to Find Documents Relevant to the Query  

```python  
# Use Ensemble Retriever for retrieval  
query = "How did Manchester United players perform in their 3-0 victory over West Ham?"  
docs = ensemble_retriever.invoke(query)  

for doc in docs:  
    print(f"Document ID: {doc.lc_id}")  
    print(f"Content: {doc.page_content}")  
    print("\n---\n")  
```  

Retrieved documents:  

![alt text](picture/image-5.png)  

![alt text](picture/image-6.png)  

##### (5) Combine Hybrid Retrieval Results with the Question and Generate a `response`  

```python  
# Combine retrieved document content into a string and pass it to query_engine  
doc_contents = "\n".join([doc.page_content for doc in docs])  

combined_query = f"My question is: {query}. I know the following information: {doc_contents}.  
                                Please answer based on this content."  

# Use query_engine for summarization and response generation  
response = query_engine.query(combined_query)  

# Print the result  
print(response)  
```  

Query text generation process:  
![alt text](picture/image-7.png)  

Returned `response`:  

![alt text](picture/image-8.png)  

> In Manchester United's 3-0 victory over West Ham, the players performed well and secured the win, with Rashford, Højlund, and McTominay scoring goals. Additionally, Rashford's performance was remarkable, as he either scored or assisted in his last four Premier League matches. Højlund was also in good form, contributing to six goals in five games across competitions. Despite some defensive issues, the team managed to field a strong lineup, with key players returning, and achieved a crucial win. In this match, West Ham players did not perform well, as it mentions that the actual reverse match at London Stadium saw West Ham win 2-0.

#### 3. `HyDE` Query Rewriting  

The core idea behind the `HyDE` principle is to optimize query representation by generating hypothetical documents, thereby improving the relevance and accuracy of retrieval results.  

![alt text](picture/image-11.png)  

Implement query rewriting optimization using the `HyDE` model:  

```python  
# Results of HyDE query rewriting  
from llama_index.core.indices.query.query_transform import HyDEQueryTransform  
from llama_index.core.query_engine import TransformQueryEngine  

query_engine = index_bge_M3.as_chat_engine(verbose=True)  

# hyde query rewriting  
hyde = HyDEQueryTransform(include_original=True)  
hyde_query_engine = TransformQueryEngine(query_engine, hyde)  
response = hyde_query_engine.query("How did Manchester United players perform in their 3-0 victory over West Ham?")  
```  

Comparison of `response` with and without query rewriting:  

![alt text](picture/image-9.png)  

Specific `response` content seems not to show improvement.  

【Base Query】  
> Manchester United players performed well in their 3-0 victory over West Ham. Rashford and Højlund scored early, and McTominay added a third goal as a substitute in the second half. Additionally, Kobbie Mainoo scored in injury time to secure the win.  

【HyDE Query】  
> Manchester United players performed well in their 3-0 victory over West Ham, controlling the match and creating opportunities. Rashford and Højlund scored early. Despite some defensive issues, they comfortably secured the win.

### IV. Model Optimization

#### 1. Two Different HuggingFace Embedding Models

##### (1) `bge-large-zh-v1.5`

![alt text](picture/image-2.png)  
![alt text](picture/image-3.png)  

Define `index_bge_large_zh` as the index name using this embedding model.

```python
# bge-large-zh-v1.5
print("bge-large-zh-v1.5:")
bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
index_bge_large_zh = VectorStoreIndex.from_documents(
    documents=documents,
    embedding=bge_embeddings,
    show_progress=1
)
```

Sample `response` generation:  

##### (2) `bge-M3`

`BGE-M3` is the first semantic vector model to integrate three major features: Multi-Linguality, Multi-Granularity, and Multi-Functionality. It significantly enhances real-world usability and has been fully open-sourced to the community.

Define `index_bge_M3` as the index name using this embedding model.

```python
# bge-M3
print("bge-M3:")
bgeM3_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-M3")
index_bge_M3 = VectorStoreIndex.from_documents(
    documents=documents,
    embedding=bgeM3_embeddings,
    show_progress=1
)
```

Sample `response` generation shows that when asked, "What model are you?" the enhanced engine incorporates football-related domain knowledge.  
![alt text](picture/image-4.png)

#### 2. Hybrid Retrieval

##### (1) Preprocessing the Document Set

Convert `documents` into a format compatible with the `.from_texts()` initialization method of `BM25Retriever` and `FAISS`.

```python
# Define an empty text list
doc_texts = []
splitted_texts = []

# Extract text content and metadata from each document
for i, doc in enumerate(documents):
    text = doc.text  # Get text content
    doc_texts.append(text if text else "")

# Split each text based on "http"
for text in doc_texts:
    splitted_texts.extend(text.split("http"))

# Create metadata for each document
doc_metadatas = [{"source": i} for i in range(len(splitted_texts))]
```

##### (2) Defining Sparse and Dense Retrieval with `BM25Retriever` and `FAISS`

```python
# Hybrid retrieval setup
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Initialize BM25 Retriever
bm25_retriever = BM25Retriever.from_texts(splitted_texts, metadatas=doc_metadatas)
bm25_retriever.k = 3

# Initialize FAISS Retriever
embedding = OpenAIEmbeddings()
faiss_vectorstore = FAISS.from_texts(splitted_texts, embedding, metadatas=doc_metadatas)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})
```

##### (3) Defining the `EnsembleRetriever`

```python
# Ensemble Retrieval
from langchain.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)
```

##### (4) Using Hybrid Retrieval to Find Relevant Documents

```python
# Use Ensemble Retriever for query
query = "How did Manchester United players perform in their 3-0 victory over West Ham?"
docs = ensemble_retriever.invoke(query)

for doc in docs:
    print(f"Document ID: {doc.lc_id}")
    print(f"Content: {doc.page_content}\n---\n")
```

Relevant documents retrieved:  
![alt text](picture/image-5.png)  
![alt text](picture/image-6.png)  

##### (5) Combining Hybrid Retrieval Results with Query to Generate `response`

```python
# Combine retrieved content with the query
doc_contents = "\n".join([doc.page_content for doc in docs])

combined_query = f"My question is: {query}. Based on the following information: {doc_contents}, please answer."

# Generate summary and response using query engine
response = query_engine.query(combined_query)
print(response)
```

Query-to-text generation process:  
![alt text](picture/image-7.png)  

Generated `response`:  
![alt text](picture/image-8.png)

---

#### 3. `HyDE` Query Rewriting

The core idea of `HyDE` (Hypothetical Document Embeddings) is to optimize query representation by generating hypothetical documents, improving the relevance and accuracy of retrieval results.  

![alt text](picture/image-11.png)  

Implementing query rewriting with `HyDE`:

```python
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

query_engine = index_bge_M3.as_chat_engine(verbose=True)

# Apply HyDE Query Rewriting
hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine, hyde)
response = hyde_query_engine.query("How did Manchester United players perform in their 3-0 victory over West Ham?")
```

Comparison of rewritten and base query responses:  
![alt text](picture/image-9.png)  

Responses indicate limited improvement:  

**Base Query Response**:  
> Manchester United players performed well in their 3-0 victory over West Ham. Rashford, Højlund, and McTominay scored, securing the win.

**HyDE Query Response**:  
> Manchester United players performed well, controlling the match and creating opportunities. Rashford and Højlund scored early goals. Despite defensive issues, the team secured a comfortable victory.

---

#### 4. `DeepEval` Evaluation Metrics Selection

##### (1) `G-Eval`

Definition: Evaluates factual correctness of the `actual_output` against the `expected_output`.

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

correctness_metric = GEval(
    name="Correctness",
    model="gpt-3.5-turbo",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)
```

##### (2) `Answer Relevancy`

Definition: Measures the relevance of the `actual_output` compared to the provided `input`.  
Formula:  
\[
Answer\ Relevancy = \frac{\text{Number of Relevant Statements}}{\text{Total Number of Statements}}
\]

```python
from deepeval.metrics import AnswerRelevancyMetric

Relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-3.5-turbo",
    include_reason=True
)
```

##### (3) `Contextual Relevancy`

Definition: Measures the relevance of the `retrieval_context` for a given `input`.  

```python
from deepeval.metrics import ContextualRelevancyMetric

ContextualRelevancy_metric = ContextualRelevancyMetric(
    threshold=0.7,
    model="gpt-3.5-turbo",
    include_reason=0
)
```

##### (4) `Hallucination`

The official documentation describes the `Hallucination` metric as follows:

> The hallucination metric determines whether your LLM generates factually correct information by comparing the `actual_output` to the `provided context`.

Calculation formula:
$$ ​Hallucination = \frac{Number of Contradicted Contexts}{Total Number of Contexts} $$

Define the `Contextual Relevancy` matrix:

```python
from deepeval.metrics import HallucinationMetric

Hallucination_metric = HallucinationMetric(threshold=0.5, model="gpt-3.5-turbo")
```

