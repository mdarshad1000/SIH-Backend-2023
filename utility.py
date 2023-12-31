from langchain.agents import Tool, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import HumanInputRun
from dotenv import load_dotenv
import openai
import re
import os

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return the original llm_output without modification
                return_values={"output": llm_output},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input along with the original llm_output
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def search_name(query):
    # print("query is", query)
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "user",
        "content": f"""You are a very smart Legal Assitant who takes a sentence as input and based on that you generate the exact name of the 
        legal documnet (Power of Attorney, Will Testament, Sale and Purchase agreement etc.) which helps their cause. You follow brevity and your response is very small. 
        Following is the user query:\n{query}"""
        }
    ],
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    # print("Intermediate response", response["choices"][0]["message"]["content"])
    return response["choices"][0]["message"]["content"]


def generate(query):
    # print("query is", query)
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "user",
        "content": f"""You are a very smart Legal Assitant who takes the name of a legal document as an input and based on that you generate a full fledged template according to Indian law. 
          Your output is a complete document without any prefixes.  
          Following is the user query:\n{query}"""
        }
    ],
    temperature=0,
    max_tokens=3900,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    # print("Intermediate response", response["choices"][0]["message"]["content"])
    return response["choices"][0]["message"]["content"]


def parse_final_answer(final_answer):
    start = final_answer.find("Final Answer:")
    if start != -1:
        final_answer = final_answer[start+14:]
    else:
        {"Final Answer not found"}
    return final_answer


# def human_tool():
    
#     url = requests.request()
#     temp = HumanInputRun(intermediate)