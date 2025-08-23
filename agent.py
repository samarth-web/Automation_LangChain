from typing import List
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool("add_numbers")
def add_numbers(numbers) :
    """Sum numbers. Use when the user asks to add/sum values."""
    return float(sum(numbers))

@tool("sub_numbers")
def sub_numbers(numbers) :
    """Subtract numbers. Use when the user asks to subtract values."""
    val = 0
    for i in numbers:
        val = val - i
    return float(val)
@tool("multiply_numbers")
def multi_numbers(numbers) :
    """Multiply numbers. Use when the user asks to multiply values."""
    val = 1
    for i in numbers:
        val = val * i
    return float(val)

@tool("divide_numbers")
def divide_numbers(numbers) :
    """Divide numbers. Use when the user asks to divide values."""
    val = 1
    for i in numbers:
        val = i // val
    return float(val)

@tool("count_words")
def count_words(text):
    """Count whitespace-separated words in text."""
    return len(text.split())

def agent_fun():
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
    tools = [add_numbers, sub_numbers, multi_numbers, divide_numbers, count_words]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use tools when helpful."),      
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)


if __name__ == "__main__":
    agent = agent_fun()

    print("\n--- Demo 1: Math ---")
    q1 = "Please add 12, 30 and 8 for me."
    print("User:", q1)
    print("Agent:", agent.invoke({"input": q1})["output"])

    print("\n--- Demo 2: Word count ---")
    q2 = "How many words are in: 'LangChain makes tool-use easy for LLM agents'?"
    print("User:", q2)
    print("Agent:", agent.invoke({"input": q2})["output"])

    print("\n--- Demo 3: Mixed reasoning ---")
    q3 = "Add 3.5 and 6.5, then tell me how many words are in 'this is a tiny test'."
    print("User:", q3)
    print("Agent:", agent.invoke({"input": q3})["output"])

    print("\n--- Demo 4: User input ---")
    q3 = str(input("Enter your prompt: "))
    print("User:", q3)
    print("Agent:", agent.invoke({"input": q3})["output"])
