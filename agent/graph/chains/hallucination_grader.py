from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_core.language_models.chat_models import BaseChatModel


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class HallucinationChain:
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model
        structured_llm_grader = self.model.with_structured_output(GradeHallucinations)

        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts, . \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Set of facts in Portuguese: \n\n {documents} \n\n LLM generation: {generation}",
                ),
            ]
        )

        self.hallucination_grader: RunnableSequence = (
            hallucination_prompt | structured_llm_grader
        )

    def get_chain(self):
        return self.hallucination_grader
