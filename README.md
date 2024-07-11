# Agent4ai

Um Agente Conversacional capaz de auxiliar novos funcionários na empresa Tech4Humans durante o processo de integração.

## Introdução

O agente utiliza de ferramentas de LLMs e RAG, fornecendo funcionalidades para descrever com precisão informações internas da empresa, como a divisão de times, produtos, valores, visão e missão, utilizando de uma base documental, além de fornecer tutoriais e explicações sobre ferramentas internas e gerenciamento de eventos, através de pesquisas em tempo real e integração com sistema de calendário.

## Solução

Minha solução utilizou como framework principal o [LangGraph](https://github.com/langchain-ai/langgraph)🦜🕸, uma ferramenta construida em cima do Langchain que permite criar agentes com um alto nível de controle sobre seus estados, memoria, interrupção, ações e se baseia na arquitetura em grafos.

A criação do agente se deve pela implementação das ferramentas auxiliares ao LLM e a criação do grafo que gerencia como o agente irá gerar cada subresposta e resposta final.

Com essa arquitetura cada nó performa uma ação, e criamos o controle de fluxo com suas arestas de conexão. Podemos controlar o estado do agente em cada parte e como atuar sobre ele, com diferentes ferramentas e diluindo as tarefas em sub objetivos e planos de caminho.

Para uma melhor organização, os componentes do grafo foi dividido em dois diretórios principais, [chains](/agent/graph/chains/) e [nodes](/agent/graph/nodes/). Chains são uma abstrtação que o LangCahin fornece para sequencias de chamadas com um LLM, tools ou processamento de dados, assim cada chain define o prompt para o llm, a chamada de alguma ferramenta, e a estrutura de mensagem de saída do llm, padronizando a forma de operação. Os nós utilizarão de seus rescpectivos chains para invokar a sequencia de chamadas, obter o resultados e tomar alguma ação sobre eles.

![graph](/images/graph_mermaid.png)
[graph_mermaid](https://drive.google.com/file/d/1z9eoB3ERiNw3pKFK52Eh-vQQYZ20b7tc/view?usp=sharing)

### Funcionalidades
---
#### **RAG**
O conceito de Geração Aumentada de Recuperação (RAG) aprimora os resultados de um LLM ao integrar informações específicas e atualizadas, usando recuperação de dados de uma determinada base de conhecimento

No contexto do desafio a técninca de RAG foi necessaria para integrar ao LLM informações sobre a documentação da empresa que foi disponiblizada, pesquisa na web para obter respostas atualizadas das ferramentas externas e integração com o sistema de calendário.

Em meu desenvolvimento criei um agente que utiliza das ideias de 
- [Adaptive-RAG](https://arxiv.org/abs/2403.14403). Um "roteador", direciona a pergunta para diferentes abordagens de recuperação.
- [Corrective-RAG](https://arxiv.org/pdf/2401.15884.pdf). Mecanismo de fallback, plano alternativo para quando o contexto recuperado é irrelevante para solucionar a pergunta.
- [Self-RAG](https://arxiv.org/abs/2310.11511). Um processo de avaliação, em que se classifica a resposta do agente como alucinação ou fora de contexto, e corrige a resposta.

#### **Roteamento**

A primeira etapa do agente com a entrada do usuário, é interpretar a questão com o llm principal e direcionar o fluxo do processo para determinado nó. Com a ideia do Adaptive-RAG, é possivel escolher a melhor ferramenta de recuperação para cada tipo e tarefa especifica.
- [Router Node](/agent/graph/nodes/router_node.py), [Router Chain](/agent/graph/chains/router.py)

#### **Documentação**

Para o LLM conseguir utilizar as informações disponibilizadas sobre a empresa foi realizado as técnicas de Embedding junto a busca de similaridade em banco de vetores. Onde o texto do documento é dividido em multiplos tokens, diluindo o texto em pacotes menores, e estes recebem uma representação vetorial, os vetores são persistidos em um banco, e posteriormente, através de métricas de similaridade é possivel recuperar informações mais correlacionadas com o contexto.

- O Embedding, foi utilizado o modelo [embed-multilingual-v3.0](https://docs.cohere.com/docs/cohere-embed#multi-lingual-models), que cria embeddings de 1024 dimensões com um contexto com o máximo de 512 tokens.

- Para o banco de vetores, foi utilizado o [MongoDB Atlas](https://www.mongodb.com/docs/atlas/), que fornece uma ferramenta de pesquisa vetorial em campos indexados. [Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/)

![doc-rag](/images/Cohere%20Multilingual%20Model.png)
[Embedding Rag](https://drive.google.com/file/d/1DsShUQCkMza8mhoc4WZTJmLWwBkT5n4c/view?usp=sharing)

Além disso, na ideia do Corrective-RAG, após o nó de "retriever" extrair os melhores chunks dos documentos associados a questão, estes documentos são avaliados em um outro nó, julgando-os um a um se são relevantes no contexto ou não. Assim, a recuperação não se garante apenas pela busca por similaridade, mas ainda mais uma etapa de interpretação do LLM. 
Caso a recuperação se deu como totalmente irrelevante, o agente é direcionado ao web search, caso contrário ele se direciona para a geração da resposta final. 
- [Retriever Node](/agent/graph/nodes/retrieve.py)
- [Grade Documents Node](/agent/graph/nodes/grade_document.py), [Retrievel Grade Chain](/agent/graph/chains/retrieval_grader.py)

#### **Web Search**

A integração com as pesquisas na web em tempo real se deu pela ferramenta [Travily](https://tavily.com/), que já possui [integração](https://python.langchain.com/v0.2/docs/integrations/retrievers/tavily/) com o framework do LangChain.

Para otimizar o processo, antes de chamar a api de busca, um nó de planejamento usa do LLM para através da questão do usário criar 3 melhores sentenças para pesquisar. Isso permite direcionar de forma precisa a pesquisa, introduzindo a interpretabilidade da questão pelo LLM. Assim a api realiza 3 melhores buscas e retorna em documentos a url da fonte junto com seu conteúdo relevante. 
- [Research Planner Node](/agent/graph/nodes/research_plan.py), [Research Planner Chain](/agent/graph/chains/planner.py)
- [Web Search Node](/agent/graph/nodes/web_search.py)

#### **Calendário**

Para integração com sistema de calendário foi utilizado a api do google clound, que permite acessar e realizar ações com o Google Calendar. Criei uma interface que permite listar um determinado número de eventos a partir do dia atual, criar eventos e acessar informações básicas do usuário. 
- [Calendar Tool](/agent/tools/calendar_tool.py)

O nó responsável pelas ações de agenda, eventos, usa do LLM junto com uma pré coleta de informação, com os próximos 5 eventos e informações do usuario, para assim decidir entre as ações de listar, criar eventos ou somente informar algo. Com a decisão tomada, também gera os parâmetros necessários para a função escolhida, por exemplo, se foi decidido criar um evento ele cria todos os parâmetros necessários para usar o metodos de criação que a ferramenta fornece. 
- [Calendar Node](/agent/graph/nodes/calendar_node.py), [Calendar Chain](/agent/graph/chains/calendar.py)

#### **Geração  Final**

O nó final do agente é responsável por gerar a resposta final com o LLM e as informações de contexto recuperadas anteriormente, junto ao histórico de conversas.

Porém, na ideia do Self-RAG, a resposta final passa por uma avaliação, verificando se a resposta é realmente útil para o útil para a entrada do usuário. Caso seja, vá para o estado final, caso contrário é retornado para o nó de pesquisa.

A implementação está simplificada, para maior segurança e consistência, poderia haver um contador que inibisse do agente entrar em loop infinito, caso saia muito do contexto e não alcance uma boa respsota. Além disso, o escape para a pesquisa foi uma escolha rápida, por o nó ter uma boa probabilidade de obter informações, que pelo menos, parecem úteis.

- [Generate Node](/agent/graph/nodes/generate.py), [Generate Chain](/agent/graph/chains/generation.py)
- [Answer Grader](/agent/graph/chains/answer_grader.py)

A construção do agente final pode ser encontrado em [agent](/agent/graph/agent.py)

## Uso

### Configuração

1. **Instalação**

    Primeiramente clone o repositorio:
    ```bash
    git clone https://github.com/samuellimabraz/Agent4ai.git
    ```
2. **Dependências**
 
    É necessario um abiente com [Python 3.10+](https://www.python.org/downloads/release/python-3100/) e realizar a instalação das dependencias que se encontra no arquivo de [requerimentos](requirements.txt):
    ```bash
    pip install -r requirements.txt
    ```
3. **Chaves de API**

    O projeto utiliza diferentes ferramentas que requerem de tokens de validação para acessa-lás.
    - **[Groq](https://groq.com/)**: Plataforma que oferece a inferência rápida em nuvem com LPUs do LLM principal.
    - **[Cohere](https://dashboard.cohere.com/)**: Fornece o modelo de Embedding utilizado para ferramenta de retriever.
    - **[Tavily](https://tavily.com/)**: Ferramenta de web search otimizada para RAG
    - **[MongoDB](https://www.mongodb.com/docs/manual/reference/connection-string/#find-your-mongodb-atlas-connection-string)**: Banco de dados utilizado para consulta de vetores e armazenamento do histórico de conversa.
    - **[Google Clound](https://developers.google.com/calendar/api/quickstart/python)**: Para ter acesso as funcionalidades do Google Calendar, mas também o ID da conta é utilizado para o gerenciamento de memória da conversa. Para simplificação, esse credenciamento é obrigatorio para inicialização do agente. Siga as intruções do link para configurar a conta.

    Ao final, crie um arquivo ```.env``` com as chaves obtidas, ficará semelhante a isso:
    ```
    GROQ_API_KEY=...
    COHERE_API_KEY=...
    TAVILY_API_KEY=..
    MONGODB_CONNECTION_STRING=...
    ```

---
### Execução

- Vá para o diretório principal:
    ```
    cd Agent4ai
    ```

- O agente foi disponibilizado através de uma API, com uso da [FastAPI](https://fastapi.tiangolo.com/) e uma interace gráfica com [Gradio](https://www.gradio.app/guides/quickstart).

    Para executar o projeto, utilize:

    ```bash
    uvicorn app:app 
    ```

    Assim incializará a API, que fornece operações de POST para envio da mensagem de requisição e operação para limpeza do histórico de conversa. A documentação é disponibilizada com Swagger, que pode ser acessada em /docs

- Para inciar a interface gráfica use:

    ```bash
    python .\interface\gui.py
    ```
Demo:

[demo](https://drive.google.com/file/d/1BkTkj2FdIEG94zlQSeOCI_VniqRPHvsP/view?usp=sharing)

## Referências 

- [DeepLearning.AI - AI Agents in LangGraph](https://learn.deeplearning.ai/accomplishments/fa462bc1-6c3c-4af0-8ba5-b5c54630fdf4?usp=sharing)
- [Advanced RAG control flow with LangGraph](https://github.com/locupleto/langgraph-course/tree/main)
- [Building a RAG Agent with LangGraph, LLaMA3–70b, and Scaling with Amazon Bedrock](https://medium.com/@philippkai/building-a-rag-agent-with-langgraph-llama3-70b-and-scaling-with-amazon-bedrock-2be03fb4088b)
- [Self-Reflective RAG with LangGraph](https://blog.langchain.dev/agentic-rag-with-langgraph/)