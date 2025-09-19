from langchain_core.language_models.chat_models import BaseChatModel
from utils.state import AgentState
from langchain.prompts import PromptTemplate

class AnswerValidation:

    def __init__(self, llm: BaseChatModel, prompt: str):
        self._llm = llm
        self._prompt = prompt

    def validate(self, state: AgentState) -> AgentState:
        question = state['original_question']
        context = state['context']
        answer = state['messages'][-1].content
        prompt = PromptTemplate.from_template(self._prompt).invoke({"question":question, "context": context, "answer": answer})
        response = self._llm.invoke(prompt)
        state['messages'].append(response)

        return state
    

