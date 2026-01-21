import streamlit as st
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
import uuid
import chatbot_nodes
from streamlit_float import float_init, float_parent, float_css_helper

st.set_page_config(
    page_title="MB Banking Chatbot PoC",
    page_icon="🤖",
    layout="wide"
)

float_init()

memory = MemorySaver()

def reset_conversation():
    memory = MemorySaver()
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.graph = chatbot_nodes.graph_builder.compile(checkpointer=memory)
    st.session_state.thread = {"configurable": {
        "thread_id": st.session_state.thread_id
    }}
    st.session_state.follow_up = False
    st.session_state.status = False
    st.session_state.status_message = ""

col1, col2 = st.columns([0.05, 0.95])
with col1:
    st.image("langgraph/resources/images/ocbc_logo.png", width = 50)
with col2:
    st.subheader("AI Search PoC Demo", anchor = False)

if "messages" not in st.session_state:
    reset_conversation()

if st.session_state.status:
    print("printing status")
    st.status(st.session_state.status_message)

def streaming_helper(events):
    for event in events:
        print(event)
        response = event["partial_response"]
        if response == " \n" or response == "\n":
            yield " \n"
        else:
            yield response

tab_poc, tab_description, tab_settings = st.tabs(["PoC Demo", "Description", "Settings"])
with tab_description:
    st.subheader("Description", divider= True)

with tab_settings:
    st.subheader("Settings", divider= True)

with tab_poc:
    have_input = False
    st.button('Start a new chat!', on_click= reset_conversation)
    # To display history of messages on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if st.session_state.status:
        print("Status as below ===")
        st.status(st.session_state.status_message)
    with st.container():
        input_css = float_css_helper(
            bottom="1rem",
            left="5%",
            right="5%",
            width="90%",
            transition=0,
            aligh="center"
        )
        float_parent(css=input_css)
        if prompt := st.chat_input("Ask your questions here!"):
            have_input = True
            st.session_state.messages.append({"role": "user", "content": prompt})
    
    if have_input:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Processing your query, please hold on...", expanded=True) as status:
                # add this for follow up clarification qns when added
                # if st.session_state.follow_up:
                #     events = chatbot_nodes
                events = chatbot_nodes.customer_query(
                    st.session_state.graph,
                    st.session_state.messages,
                    prompt,
                    st.session_state.thread_id,
                )
                response_chunks = []
                response_placeholder = None
                button_placeholder = None
                for event in events:
                    if event.get("router"):
                        status.update(label="Understanding your query and finding the best way to serve you...", state="running")
                    elif event.get("is_jailbreak"):
                        status.update(label="Guard activated! Your query has been deemed as malicious ⚠️⚠️⚠️", state="complete")
                    elif event.get("in_app"):
                        status.update(label="Your query seems to relate to a product or service within the app! Finding out more...", state="running")
                    elif event.get("faq"):
                        status.update(label="Looking up relevant information for your enquiry...", state="running")
                    elif event.get("generate_final_response"):
                        status.update(label="Generating response...", state="running")
                        response_placeholder = st.empty()
                    elif "partial_response" in event:
                        response_chunks.append(event["partial_response"])
                        if response_placeholder is not None:
                            response_placeholder.markdown("".join(response_chunks))
                    elif event.get("deeplink"):
                        button_placeholder = st.empty()
                        if button_placeholder is None:
                            button_placeholder = st.empty()
                        label = "Open in app"
                        if event.get("product_or_service"):
                            label = f"Open {event['product_or_service']}"
                        button_placeholder.link_button(label, event["deeplink"])
                    else:
                        print(event)

                if response_chunks:
                    result = "".join(response_chunks)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    status.update(label="Response successfully generated!", state="complete")
