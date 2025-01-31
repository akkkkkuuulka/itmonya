import os
import json
import re
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent, Tool

from agent.model import load_model
from agent.tools.search_generator import SearchQueryGeneratorTool
from agent.prompts.react_ru import REACT_RU_JSON_PROMPT

llm = load_model()
wiki = WikipediaAPIWrapper(top_k_results=3)  
search_gen_tool = SearchQueryGeneratorTool()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wiki.run,
    description="""Useful for:
    - General knowledge questions
    - Historical facts
    - Scientific concepts
    - Biographical information
    - Cultural references"""
)

tools = [
    search_gen_tool,
    TavilySearchResults(max_results=3),  # Web search
    # wikipedia_tool             # Wikipedia search
]

agent = create_react_agent(llm, tools, REACT_RU_JSON_PROMPT)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

async def create_async_agent():
    tools = [SearchQueryGenerator(), TavilySearchResults()]
    
    agent = create_react_agent(
        llm=ChatGoogleGenerativeAI(model=load_model()),
        tools=tools,
        prompt=REACT_PROMPT
    )
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )


def process_query(input_question, query_id):
    query = input_question
    
    try:
        result = agent_executor.invoke({"input": query})
        answer_text = result["output"]
        intermediate_steps = result["intermediate_steps"]
    except Exception as e:
        return json.dumps({
            "id": query_id,
            "error": f"Agent execution failed: {str(e)}"
        })

    sources = []
    for step in intermediate_steps:
        action, observation = step
        if action.tool == "tavily_search_results_json":
            for item in observation:
                if 'url' in item:
                    sources.append(item['url'])
        elif action.tool == "Wikipedia":
            if "Sources: " in observation:
                urls = observation.split("Sources: ")[1].split(", ")
                sources.extend(urls)

    sources = list(set(sources))[:3]
    answer_parsed = json.loads(answer_text)
    return json.dumps({
        "id": query_id,
        "answer": answer_parsed['answer_number'],
        "reasoning": answer_parsed['reasoning'],
        "sources": sources
    }, ensure_ascii=False)
