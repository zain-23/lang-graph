from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from chains import generation_chain, reflection_chain

# Create a message graph for the reflection system
graph = MessageGraph()
GENERATE = "generate"
REFLECT = "reflect"

def generate_node(state):
    return generation_chain.invoke({
        "messages": state
    })

def reflect_node(state):
    response = reflection_chain.invoke({
        "messages": state
    })
    return [HumanMessage(content=response.content)]

# Add nodes to the graph
graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

# Entry point for the graph
graph.set_entry_point(GENERATE)

# Create a should continue function to determine if the graph should continue
def should_continue(state):
    print("PRINT")
    if(len(state) > 2):
        return END
    return REFLECT

# Connect edges in the graph
graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()
response = app.invoke(HumanMessage(content="AI agents taking over content creation"))
print(response)