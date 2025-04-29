# Agent4ai

A Conversational Agent capable of assisting new employees at the Tech4Humans company during the onboarding process.

## Introduction

The agent uses LLMs and RAG tools, providing functionalities to accurately describe internal company information, such as team divisions, products, values, vision, and mission, using a document base, as well as providing tutorials and explanations about internal tools and event management, through real-time searches and integration with a calendar system.

## Solution

My solution used [LangGraph](https://github.com/langchain-ai/langgraph)🦜🕸 as the main framework, a tool built on top of Langchain that allows creating agents with a high level of control over their states, memory, interruption, actions, and is based on graph architecture.

The agent's creation involves implementing auxiliary tools for the LLM and creating the graph that manages how the agent will generate each sub-response and the final response.

With this architecture, each node performs an action, and we create the control flow with its connecting edges. We can control the agent's state in each part and how to act on it, with different tools and breaking down tasks into sub-objectives and path plans.

For better organization, the graph components were divided into two main directories, [chains](/agent/graph/chains/) and [nodes](/agent/graph/nodes/). Chains are an abstraction that LangChain provides for sequences of calls with an LLM, tools, or data processing. Thus, each chain defines the prompt for the LLM, the call to some tool, and the structure of the LLM's output message, standardizing the mode of operation. The nodes will use their respective chains to invoke the sequence of calls, obtain the results, and take some action on them.

![graph](/images/graph_mermaid.png)
[graph_mermaid](https://drive.google.com/file/d/1z9eoB3ERiNw3pKFK52Eh-vQQYZ20b7tc/view?usp=sharing)

### Features
---
#### **RAG**
The concept of Retrieval-Augmented Generation (RAG) enhances the results of an LLM by integrating specific and updated information, using data retrieval from a given knowledge base.

In the context of the challenge, the RAG technique was necessary to integrate information about the company's documentation provided to the LLM, web search to obtain updated answers from external tools, and integration with the calendar system.

In my development, I created an agent that uses the ideas of:
- [Adaptive-RAG](https://arxiv.org/abs/2403.14403). A "router" directs the query to different retrieval approaches.
- [Corrective-RAG](https://arxiv.org/pdf/2401.15884.pdf). Fallback mechanism, an alternative plan for when the retrieved context is irrelevant to solving the question.
- [Self-RAG](https://arxiv.org/abs/2310.11511). An evaluation process, where the agent's response is classified as hallucination or out of context, and the response is corrected.

#### **Routing**

The first step of the agent with user input is to interpret the question with the main LLM and direct the process flow to a specific node. With the idea of Adaptive-RAG, it is possible to choose the best retrieval tool for each specific type and task.
- [Router Node](/agent/graph/nodes/router_node.py), [Router Chain](/agent/graph/chains/router.py)

#### **Documentation**

For the LLM to use the information provided about the company, Embedding techniques were used along with similarity search in a vector database. The document text is divided into multiple tokens, breaking the text into smaller packages, and these receive a vector representation. The vectors are persisted in a database, and subsequently, through similarity metrics, it is possible to retrieve information more correlated with the context.

- For Embedding, the [embed-multilingual-v3.0](https://docs.cohere.com/docs/cohere-embed#multi-lingual-models) model was used, which creates 1024-dimensional embeddings with a maximum context of 512 tokens.

- For the vector database, [MongoDB Atlas](https://www.mongodb.com/docs/atlas/) was used, which provides a vector search tool on indexed fields. [Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/)

![doc-rag](/images/Cohere%20Multilingual%20Model.png)
[Embedding Rag](https://drive.google.com/file/d/1DsShUQCkMza8mhoc4WZTJmLWwBkT5n4c/view?usp=sharing)

Furthermore, following the idea of Corrective-RAG, after the "retriever" node extracts the best chunks from the documents associated with the question, these documents are evaluated in another node, judging them one by one whether they are relevant in the context or not. Thus, the retrieval is not guaranteed only by the similarity search but also by an additional interpretation step by the LLM.
If the retrieval is deemed totally irrelevant, the agent is directed to the web search; otherwise, it proceeds to generate the final response.
- [Retriever Node](/agent/graph/nodes/retrieve.py)
- [Grade Documents Node](/agent/graph/nodes/grade_document.py), [Retrieval Grade Chain](/agent/graph/chains/retrieval_grader.py)

#### **Web Search**

Integration with real-time web searches was done using the [Tavily](https://tavily.com/) tool, which already has [integration](https://python.langchain.com/v0.2/docs/integrations/retrievers/tavily/) with the LangChain framework.

To optimize the process, before calling the search API, a planning node uses the LLM to create the 3 best sentences to search based on the user's question. This allows directing the search precisely, introducing the LLM's interpretability of the question. Thus, the API performs the 3 best searches and returns the source URL along with its relevant content in documents.
- [Research Planner Node](/agent/graph/nodes/research_plan.py), [Research Planner Chain](/agent/graph/chains/planner.py)
- [Web Search Node](/agent/graph/nodes/web_search.py)

#### **Calendar**

Integration with the calendar system used the Google Cloud API, which allows accessing and performing actions with Google Calendar. I created an interface that allows listing a certain number of events starting from the current day, creating events, and accessing basic user information.
- [Calendar Tool](/agent/tools/calendar_tool.py)

The node responsible for agenda actions, events, uses the LLM along with pre-collected information (next 5 events and user info) to decide between listing, creating events, or just providing information. Once the decision is made, it also generates the necessary parameters for the chosen function. For example, if it decided to create an event, it creates all the parameters needed to use the creation method provided by the tool.
- [Calendar Node](/agent/graph/nodes/calendar_node.py), [Calendar Chain](/agent/graph/chains/calendar.py)

#### **Final Generation**

The final node of the agent is responsible for generating the final response with the LLM and the previously retrieved context information, along with the conversation history.

However, in the idea of Self-RAG, the final response undergoes an evaluation, checking if the response is truly useful for the user's input. If it is, proceed to the final state; otherwise, it returns to the research node.

The implementation is simplified. For greater security and consistency, there could be a counter to prevent the agent from entering an infinite loop if it goes too far off context and does not achieve a good response. Additionally, the escape to the search node was a quick choice, as the node has a good probability of obtaining information that, at least, seems useful.

- [Generate Node](/agent/graph/nodes/generate.py), [Generate Chain](/agent/graph/chains/generation.py)
- [Answer Grader](/agent/graph/chains/answer_grader.py)

The construction of the final agent can be found in [agent](/agent/graph/agent.py)

## Usage

### Setup

1.  **Installation**

    First, clone the repository:
    ```bash
    git clone https://github.com/samuellimabraz/Agent4ai.git
    ```
2.  **Dependencies**

    An environment with [Python 3.10+](https://www.python.org/downloads/release/python-3100/) is required. Install the dependencies found in the [requirements](requirements.txt) file:
    ```bash
    pip install -r requirements.txt
    ```
3.  **API Keys**

    The project uses different tools that require validation tokens to access them.
    - **[Groq](https://groq.com/)**: Platform offering fast cloud inference with LPUs for the main LLM.
    - **[Cohere](https://dashboard.cohere.com/)**: Provides the Embedding model used for the retriever tool.
    - **[Tavily](https://tavily.com/)**: Web search tool optimized for RAG.
    - **[MongoDB](https://www.mongodb.com/docs/manual/reference/connection-string/#find-your-mongodb-atlas-connection-string)**: Database used for vector queries and storing conversation history.
    - **[Google Cloud](https://developers.google.com/calendar/api/quickstart/python)**: To access Google Calendar functionalities, but the account ID is also used for conversation memory management. For simplification, this credentialing is mandatory for agent initialization. Follow the link's instructions to configure the account.

    Finally, create a `.env` file with the obtained keys. It will look similar to this:
    ```
    GROQ_API_KEY=...
    COHERE_API_KEY=...
    TAVILY_API_KEY=...
    MONGODB_CONNECTION_STRING=...
    ```

---
### Execution

- Go to the main directory:
    ```
    cd Agent4ai
    ```

- The agent has been made available through an API, using [FastAPI](https://fastapi.tiangolo.com/) and a graphical interface with [Gradio](https://www.gradio.app/guides/quickstart).

    To run the project, use:

    ```bash
    uvicorn app:app
    ```

    This will initialize the API, which provides POST operations for sending request messages and an operation for clearing the conversation history. Documentation is available via Swagger, accessible at /docs

- To start the graphical interface use:

    ```bash
    python ./interface/gui.py
    ```
Demo:

[demo](https://drive.google.com/file/d/1BkTkj2FdIEG94zlQSeOCI_VniqRPHvsP/view?usp=sharing)

## References

- [DeepLearning.AI - AI Agents in LangGraph](https://learn.deeplearning.ai/accomplishments/fa462bc1-6c3c-4af0-8ba5-b5c54630fdf4?usp=sharing)
- [Advanced RAG control flow with LangGraph](https://github.com/locupleto/langgraph-course/tree/main)
- [Building a RAG Agent with LangGraph, LLaMA3–70b, and Scaling with Amazon Bedrock](https://medium.com/@philippkai/building-a-rag-agent-with-langgraph-llama3-70b-and-scaling-with-amazon-bedrock-2be03fb4088b)
- [Self-Reflective RAG with LangGraph](https://blog.langchain.dev/agentic-rag-with-langgraph/)