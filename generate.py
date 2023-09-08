from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain import OpenAI, LLMChain
from langchain.tools import HumanInputRun
from utility import CustomOutputParser, CustomPromptTemplate, search, generate
from langchain.memory import ConversationBufferWindowMemory
import os
from dotenv import load_dotenv

load_dotenv()

# Set up the base template
template_with_history = """You are a Legal Assistant whose task is to generate a personalised canonical and high quality legal document.
You need to generate the legal document with the user's detail which is used to address the user's need. Ask targeted questions from the user.
You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must understand and eventually generate a legal document for it.
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer is a full fledged legal document and nothing else.

Begin! Remember to only generate Legal Documents which the user has asked. You should ask human for their details which you need while generating the legal document.

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}
"""    

output_parser = CustomOutputParser()

human_tool = HumanInputRun()

# Define which tools the agent can use to answer user queries
tools = [
    Tool(
        name = "Search",
        func=search,
        description="useful for when you need to look for the name of legal documents based on the user input."
    ),
    Tool(
        name = "Ask Human",
        func = human_tool.run,
        description = "useful only for when you have obtained the entire template of the legal document and want to fill the unknown values so take input from human"
    ),
    Tool(
        name = "Get Template",
        func = generate,
        description = "useful for when you need to get a legal document template."
    )
]


prompt_with_history = CustomPromptTemplate(
    template=template_with_history,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "history"]
)


llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", openai_api_key=os.getenv('OPENAI_API_KEY'))
llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)


memory=ConversationBufferWindowMemory(k=4)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

agent_executor.run()

