# graph.py
"""
Connects the nodes into a LangGraph flow.
For now: simple decision → route question to docs/advice/marketing.
"""

from langgraph.graph import StateGraph, END
from nodes import docs_node, advice_node, marketing_node
from pydantic import BaseModel


print("[graph.py] Building graph...")

# Define state schema
class State(BaseModel):
    question: str
    answer: str
    source: str

# Graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("docs", docs_node)
workflow.add_node("advice", advice_node)
workflow.add_node("marketing", marketing_node)

# Routing function for decision logic
def decide_next_node(state: State) -> str:
    q = state.question.lower()
    if any(word in q for word in ["document", "plan", "roadmap", "policy", "investor",
        "about", "what do we", "what is the startup", "help with",
        "product strategy", "internal", "services", "ai assistants", "revenue model", "key functions"]):
        return "docs"
    elif any(word in q for word in ["market", "campaign", "marketing", "post","customer", "idea", "audience"]):
        return "marketing"
    else:
        return "advice"

# LangGraph-compatible router node
def router_node(state: State) -> dict:
    next_node = decide_next_node(state)
    return {"next": next_node}


workflow.add_node("router", router_node)

# Add router as a conditional node
workflow.add_conditional_edges("router", decide_next_node, {
    "docs": "docs",
    "advice": "advice",
    "marketing": "marketing"
})

# Set entry point to the router node name (not the function itself)
workflow.set_entry_point("router")


# Connect nodes → all go to END
workflow.add_edge("docs", END)
workflow.add_edge("advice", END)
workflow.add_edge("marketing", END)

app_graph = workflow.compile()

print("[graph.py] Graph ready.")
