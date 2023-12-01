from typing import Dict, List, Any
from pydantic import BaseModel, Field
from langchain.llms import BaseLLM
from langchain.chains.base import Chain
from langchain.callbacks import get_openai_callback
from deepening_stages import DEEPENING_CONVERSATION_STAGES
from deepening_stage_analyzer_conversation import (
    DeepeningSalesConversationChain,
    DeepeningStageAnalyzerChain,
)


class DeepeningSalesGPT(Chain, BaseModel):
    """Controller model for the Sales Agent."""

    deepening_conversation_history: List[str] = []
    current_conversation_stage: str = DEEPENING_CONVERSATION_STAGES.get("1")
    deepening_stage_analyzer_chain: DeepeningStageAnalyzerChain = Field(...)
    deepening_sales_conversation_utterance_chain: DeepeningSalesConversationChain = Field()
    conversation_stage_dict: Dict = DEEPENING_CONVERSATION_STAGES

    salesperson_name: str = "David"
    salesperson_role: str = "Sales executive"
    company_name: str = "Smart Glass Country"
    company_business: str = "We sell smart glass and smart film"
    company_values: str = "We Make Ordinary Glass Extraordinary"
    conversation_purpose: str = "The purpose of this conversation is that you need to discover what user wants and go deeper and ask question related to it and after that handover the chat"
    conversation_type: str = "chat"

    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, "1")

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def seed_agent(self):
        # Step 1: seed the conversation
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.deepening_conversation_history = []

    def determine_conversation_stage(self):
        with get_openai_callback() as cb:
            conversation_stage_id = self.deepening_stage_analyzer_chain.run(
                deepening_conversation_history='"\n"'.join(self.deepening_conversation_history),
                conversation_stage_dict = self.conversation_stage_dict,
                current_conversation_stage=self.current_conversation_stage,
            )

        self.current_conversation_stage = self.retrieve_conversation_stage(
            conversation_stage_id
        )

        # print(f"\n<Conversation Stage>: {self.current_conversation_stage}\n")
        return conversation_stage_id, cb.total_tokens, cb.total_cost

    def human_step(self, human_input):
        # process human input
        human_input = human_input + "<END_OF_TURN>"
        self.deepening_conversation_history.append(human_input)


    def step(self):
        res_step = self._call(inputs={})
        return res_step

    def _call(self, inputs: Dict[str, Any]) -> None:
        """Run one step of the sales agent."""

        # Generate agent's utterance
        with get_openai_callback() as cb:
            ai_message = self.deepening_sales_conversation_utterance_chain.run(
                salesperson_name=self.salesperson_name,
                salesperson_role=self.salesperson_role,
                company_name=self.company_name,
                company_business=self.company_business,
                company_values=self.company_values,
                conversation_purpose=self.conversation_purpose,
                deepening_conversation_history="\n".join(self.deepening_conversation_history),
                conversation_stage=self.current_conversation_stage,
                conversation_type=self.conversation_type,
            )

        # Add agent's response to conversation history
        self.deepening_conversation_history.append(ai_message)
        res = ai_message.rstrip("<END_OF_TURN>")
        return res,cb.total_tokens, cb.total_cost

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "DeepeningSalesGPT":
        """Initialize the DeepeningSalesGPT Controller."""
        deepening_stage_analyzer_chain = DeepeningStageAnalyzerChain.from_llm(llm, verbose=verbose)
        deepening_sales_conversation_utterance_chain = DeepeningSalesConversationChain.from_llm(
            llm, verbose=verbose
        )

        return cls(
            deepening_stage_analyzer_chain=deepening_stage_analyzer_chain,
            deepening_sales_conversation_utterance_chain=deepening_sales_conversation_utterance_chain,
            verbose=verbose,
            **kwargs,
        )
