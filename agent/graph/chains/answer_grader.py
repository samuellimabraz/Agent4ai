from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_core.language_models.chat_models import BaseChatModel


class GradeAnswer(BaseModel):

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class GradeAnswerChain:
    def __init__(self, model: BaseChatModel):
        self.model = model
        structured_llm_grader = self.model.with_structured_output(GradeAnswer)

        system = """You are an evaluator who evaluates whether a generated response is well written and can be a good response to user input \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer is good."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "User input: \n\n {question} \n\n LLM generation: {generation}",
                ),
            ]
        )

        self.answer_grader: RunnableSequence = answer_prompt | structured_llm_grader

    def get_chain(self):
        return self.answer_grader
