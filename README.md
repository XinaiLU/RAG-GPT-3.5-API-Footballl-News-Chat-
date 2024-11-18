## <center>Introduction to Information Retrieval - Final Project</center>
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

![alt text](<未命名文件 (2).png>)

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

![alt text](image-1.png)

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

![alt text](8abea2cac30f25fef7fb7a50b756e3e.png)
```
