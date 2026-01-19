from llm_utils import create_llm_instance

from typing import Sequence, List, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path
from typing_extensions import TypedDict, Literal, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt, StreamWriter

# import json, requests
import re
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from guardrails import Guard


# Define embedding model first, we will use bge-en-large-v1.5 as before

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
COLLECTION_NAME = "poc_collection"
PERSIST_DIR = (Path(__file__).resolve().parent.parent / "chromadb" / "data").resolve()

llm_model = "gpt-5-mini"

# Placeholder here first, will need to get data from Jon and reformat
available_products_services = """
    1. Precious Metals
    2. Unit Trust
    3. Insurance 
    4. Dispute Card Transactions
    5. Money Lock
"""
deeplink_list = {
    "Precious Metals" : "https://internet.ocbc.com/internet-banking/digital/iis/PublicPushNotification/Redirect?code=pm-buysell",
    "Unit Trust" : "https://internet.ocbc.com/internet-banking/digital/iis/PublicPushNotification/Redirect?code=ut-products",
    "Insurance" : "https://internet.ocbc.com/internet-banking/digital/iis/PublicPushNotification/Redirect?code=banca",
    "Dispute Card Transactions" : "https://internet.ocbc.com/internet-banking/digital/iis/PublicPushNotification/Redirect?code=dispute",
    "Money Lock": "https://internet.ocbc.com/internet-banking/digital/iis/PublicPushNotification/Redirect?code=moneylock"
}

class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self) -> None:
        self._model = SentenceTransformer(EMBEDDING_MODEL)

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(sentences=input)
    
embedding_fn = MyEmbeddingFunction()

# Call chromadb in so we can query that local vector store
client = chromadb.PersistentClient(path=str(PERSIST_DIR))
collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
)

# Retrieval strat here, where the query is formatted, embedded and used to search the chromadb locally, 
# Employ a hybrid search strategy of dense vector search + compute bm25 scores for query to candidate docs
# score dfusion via normalization and weighted blend of 0.65 dense normalized, 0.35 bm25 normalized.
_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")

def _format_query(query: str) -> str:
    return " ".join(query.strip().split())

def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())

def _bm25_scores(query: str, docs: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    if not docs:
        return []
    tokenized_docs = [_tokenize(doc) for doc in docs]
    doc_lens = [len(tokens) for tokens in tokenized_docs]
    avgdl = sum(doc_lens) / max(len(doc_lens), 1)
    df: Dict[str, int] = {}
    for tokens in tokenized_docs:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    query_tokens = _tokenize(query)
    scores = [0.0] * len(docs)
    n_docs = len(docs)
    for term in query_tokens:
        term_df = df.get(term, 0)
        if term_df == 0:
            continue
        idf = max(0.0, (n_docs - term_df + 0.5) / (term_df + 0.5))
        for i, tokens in enumerate(tokenized_docs):
            tf = tokens.count(term)
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * (doc_lens[i] / (avgdl or 1.0)))
            scores[i] += idf * (tf * (k1 + 1)) / (denom or 1.0)
    return scores

def retrieve_documents(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    formatted = _format_query(query)
    if not formatted:
        return []
    query_embedding = embedding_fn([formatted])[0]
    candidate_k = max(n_results * 4, 20)
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = dense_results.get("documents", [[]])[0]
    if not docs:
        return []
    bm25_scores = _bm25_scores(formatted, docs)
    distances = dense_results.get("distances", [[]])[0]
    dense_scores = [1.0 / (1.0 + d) for d in distances]
    max_dense = max(dense_scores) if dense_scores else 1.0
    max_bm25 = max(bm25_scores) if bm25_scores else 1.0
    weight_bm25 = 0.35
    combined = []
    ids = dense_results.get("ids", [[]])[0]
    metas = dense_results.get("metadatas", [[]])[0]
    for i in range(len(docs)):
        dense_norm = dense_scores[i] / max_dense if max_dense else 0.0
        bm25_norm = bm25_scores[i] / max_bm25 if max_bm25 else 0.0
        score = (1 - weight_bm25) * dense_norm + weight_bm25 * bm25_norm
        combined.append(
            {
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i] if metas else None,
                "score": score,
            }
        )
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:n_results]

# Define memory state to persist (what states to store here to persist)
class State(TypedDict):
    msg_history: Annotated[Sequence[BaseMessage], add_messages]
    # classification of query: In-App Product/Svc or not
    in_app: bool
    # number of times customer keeps requerying for clarification -> goto fallback node for cust svc
    cust_tries:int = Field(default=0)
    # responses generated
    response:str = Field(default="")
    # if query requires additional clarification
    follow_up:str

class JailbreakCheck(BaseModel):
    is_jailbreak: bool = Field(description="True if prompt is jailbreak attempt")
    risk: float = Field(ge=0.0, le=1.0)
    reason: str

# Define BaseModels by Pydantic for response output and llm desc
class Router(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning behind the classification")
    in_app: bool = Field(
        ...,
        description="True if the query asks about in-app products/services; false otherwise.",
    )

class Prod_Services_Query(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning behind the classification")
    product_or_service: str = Field(description="Name of the matched in-app product/service, or None if not applicable.")

# Define nodes within pipeline -> at the same time, need to prompt engineer txt files and point to them

# def node_guardrail(state:State, writer:StreamWriter):
#     print("The query has been determined as trying to jailbreak")
#     writer({"jailbreak detected":"jailbreak detected"})
#     return

def node_router(state: State, writer: StreamWriter) -> Command[Literal["in_app", "faq"]]:
    writer({"router": True})
    # Router is fed as struct so LLM return structured output
    llm_router = create_llm_instance(llm_model, Router)

    # separate into its own node going forward
    # llm_guard = create_llm_instance(llm_model, JailbreakCheck)
    # guard_prompt = """
    #     Classify the user input for jailbreak risk as well as other profanities and insults.
    #     Return JSON with is_jailbreak if it is jailbreak risk or profanities and insults, risk, reason.
    #     User input: {user_input}
    # """

    # Call prompt files
    system_prompt_path = "langgraph/prompt_files/node_router_system_prompt.txt"
    user_prompt_path = "langgraph/prompt_files/node_router_user_prompt.txt"

    with open(system_prompt_path, 'r') as file:
        system_prompt = file.read()
    with open(user_prompt_path, 'r') as file:
        user_prompt = file.read()
    
    print("=== Routing Node ===")
    print(state)

    # NOTE: need to change the indexing for this later on since router node will not be at top of list. 
    user_prompt = user_prompt.format(message= state["msg_history"][-1].content)
    system_prompt = system_prompt.format(available_products_services= available_products_services)

    # separate into its own node going forward
    # guard_result = llm_guard.invoke([SystemMessage(guard_prompt.format(user_input=user_prompt))])
    # if guard_result.is_jailbreak:
    if "suicide" in user_prompt:
        writer({"is_jailbreak": True})
        goto=END
        return Command(goto=goto)

    try:
        result = llm_router.invoke([SystemMessage(system_prompt), HumanMessage(user_prompt)])
    # NOTE: Error page not implemented for PoC
    except AssertionError:
        goto="error page"
        return Command(goto=goto)
    
    print(result)
    writer({"in_app": result.in_app})
    print("=== END ROUTING NODE ===")

    if result.in_app:
        goto = "in_app"
    else:
        goto = "faq"

    return Command(goto=goto)

def node_inapp(state: State, writer: StreamWriter):

    print("=== At Inapp Prods and Svcs NODE ===")
    print("=== Starting identification of matched product or service ===")

    # Prod_Services_Query is fed as a struct
    llm_inapp = create_llm_instance(llm_model, Prod_Services_Query)

    identify_prod_service_system_prompt_path = "langgraph/prompt_files/identify_prod_service_system_prompt.txt"
    identify_prod_service_user_prompt_path = "langgraph/prompt_files/identify_prod_service_user_prompt.txt"

    with open(identify_prod_service_system_prompt_path, 'r') as file:
        system_prompt_identify = file.read()
    with open(identify_prod_service_user_prompt_path, 'r') as file:
        user_prompt_identify = file.read()

    history = ""
    for message in state["msg_history"]:
        if isinstance(message, HumanMessage):
            history = history + "<user>\n" + message.content + "\n<\\user>\n"
        else:
            history = history + "<assistant>\n" + message.content + "\n<\\assistant>\n"

    system_prompt_identify = system_prompt_identify.format(available_products_services= available_products_services)
    user_prompt_identify = user_prompt_identify.format(history= history)

    result = llm_inapp.invoke([SystemMessage(system_prompt_identify), HumanMessage(user_prompt_identify)])
    print("=== PROD OR SERVICE IDENTIFIED ===")
    print(result.product_or_service)

    # Fetch exact match of deeplink here using result.product_or_service
    if result.product_or_service:
        deeplink = deeplink_list[result.product_or_service]
    else:
        deeplink = None

    #Query the vector store with user's query
    last_user = next(
    m for m in reversed(state["msg_history"]) if isinstance(m, HumanMessage)
    )
    docs = retrieve_documents(last_user.content, 10)

    print("=== NOW GENERATE FINAL RESPONSE TO OUTPUT TO USER ===")
    writer({"generate_final_response": True})

    llm_prod_svc = create_llm_instance(llm_model)

    prod_service_generation_system_prompt = "langgraph/prompt_files/prod_service_generation_system_prompt.txt"
    with open(prod_service_generation_system_prompt, 'r') as file:
        system_prompt = file.read()

    system_prompt = system_prompt.format(history = history, product_or_service = result.product_or_service, deeplink = deeplink, retrieved_docs = docs)
    final_response_stream = llm_prod_svc.stream([SystemMessage(system_prompt)])
    final_response = ""
    for f in final_response_stream:
        writer({"partial_response": f.content.replace("$", "\\$")})
        final_response = final_response + f.content
    if deeplink:
        writer({"deeplink": deeplink, "product_or_service": result.product_or_service})

def node_faq(state: State, writer: StreamWriter):
    print("=== At FAQ NODE ===")
    writer({"faq": True})

    llm_faq = create_llm_instance(llm_model)

    faq_generation_system_prompt_path = "langgraph/prompt_files/faq_generation_system_prompt.txt"

    with open(faq_generation_system_prompt_path, 'r') as file:
        system_prompt = file.read()

    history = ""
    for message in state["msg_history"]:
        if isinstance(message, HumanMessage):
            history = history + "<user>\n" + message.content + "\n<\\user>\n"
        else:
            history = history + "<assistant>\n" + message.content + "\n<\\assistant>\n"

    #Query the vector store with user's query
    last_user = next(
    m for m in reversed(state["msg_history"]) if isinstance(m, HumanMessage)
    )
    docs = retrieve_documents(last_user.content, 10)

    system_prompt = system_prompt.format(history= history, docs=docs)

    print("=== NOW GENERATE FINAL RESPONSE TO OUTPUT TO USER ===")
    writer({"generate_final_response": True})

    final_response_stream = llm_faq.stream([SystemMessage(system_prompt)])
    final_response = ""
    for f in final_response_stream:
        writer({"partial_response": f.content.replace("$", "\\$")})
        final_response = final_response + f.content

# Create graph and build -> with main function
print("Creating Graph...")
graph_builder = StateGraph(State)
graph_builder.add_node("router", node_router)
graph_builder.add_node("in_app", node_inapp)
graph_builder.add_node("faq", node_faq)

graph_builder.add_edge(START,"router")
graph_builder.add_edge("in_app",END)
graph_builder.add_edge("faq",END)

def customer_query(chatbot, history, query:str, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    history_list = []
    for h in history[-5:-1]:
        if h["role"] == "assistant":
            history_list.append(AIMessage(content = h["content"]))
        elif h["role"] == "user":
            history_list.append(HumanMessage(content = h["content"]))
    history_list.append(HumanMessage(content=query))
    events = chatbot.stream({"msg_history": history_list, "cust_tries": 0, "follow_up": "", "in_app": False, "response": ""}, config, stream_mode="custom")
    return events

if __name__ == "__main__":
    # docs = retrieve_documents("What is price of gold?", 5)
    print("docs")
    thread_id = "1"
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    events = customer_query(graph, [], "i want to disput card txn", thread_id)
    for event in events:
        print(f"===event=== {event} ===event===")
