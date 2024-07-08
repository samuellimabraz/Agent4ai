# Agent4ai

Este repositorio contém minha solução para o desafio de IAG proposto pelo TechLab Tech4ai

## Introdução

O desafio consistia em criar um agente conversaional que fosse capaz de ajudar novos funcionarios da Tech4ai os ajudando a se integrar no time.

Resumidamente o agente deve ser capaz de responder perguntas frequentes sobre a empresa, fornecer tutorias de ferramentras internas além de se integrar com um sistema de calendario.

## Solução

Minha solução utilizou como framework principal o LangGraph, uma ferramenta construida em cima do Langchain que permite criar agentes com um alto nível de controle sobre seus estados, memoria e ações e se baseia na ideia de grafos.

Em meu desenvolvimento criei um agente que utiliza das ideias de Corrective-RAG e Self-RAG, onde o proprio agente se avalia, verificando possiveis alucinações ou respsotas mal redigidas.

A criação do agente se deve pela implementação das ferramentas auxiliares ao LLM e a criação do grafo que gerencia como o LLM irá gerar a resposta final.

### Ferramentas

#### RAG
O conveito de Geração Aumentada de Recuperação (RAG) aprimora os resultados de um LLM ao integrar informações específicas e atualizadas, sem a necessidade de alterar o modelo de IA subjacente. Essas informações podem ser mais recentes que as do LLM e adaptadas às necessidades específicas de uma organização ou setor. Como resultado, a IA generativa é capaz de fornecer respostas mais pertinentes e contextualizadas, baseadas em dados altamente atuais.

No contexto do desafio a técninca de RAG foi necessaria para integrar ao LLM informações osbre a documentação da empresa que foi disponiblizada, pesquisa na web para obter respsotas atualizadas das ferramentas externas 

#### Documentação

Para o LLM conseguir utilizar as informações dispnibilizadas sobre a empresa foi realizado as técnicas de Embedding junto a Banco de Vetores.