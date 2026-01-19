import httpx
from langchain_openai import ChatOpenAI
import json
from langchain_core.messages import HumanMessage, SystemMessage
import hashlib
from time import sleep
from openai import OpenAI

# CACHE FOR POC (if needed)
cache_path = "langgraph/cache/cache.json"
with open(cache_path, 'r') as file:
    cache = json.load(file)

def update_cache(data):
    with open(cache_path, 'w') as file:
        json.dump(data, file)

class mock_response():
    def __init__(self, content):
        self.content = content
    
# CREATE LLM INSTANCE HERE 
class llm_instance():

    def __init__(self, llm_model, struct=None, jailbreak=False, cache_toggle=True):
        self.llm_model = llm_model
        self.struct = struct
        self.jailbreak = jailbreak
        self.cache_toggle = cache_toggle
        if self.cache_toggle and self.llm_model.model_name not in cache:
            cache[self.llm_model.model_name] = {}
            update_cache(cache)
        
    def invoke(self, msg_list, *args, **kwargs):
        query = " ".join([msg.content for msg in msg_list])
        hashobj = hashlib.sha256(query.encode())
        k = str(int.from_bytes(hashobj.digest(), 'big'))

        if self.cache_toggle and self.llm_model.model_name in cache:
            if k in cache[self.llm_model.model_name]:
                print("hit cache")
                sleep(2)
                if self.struct:
                    result = json.loads(cache[self.llm_model.model_name][k])
                    return self.struct(**result)
                else:
                    return mock_response(cache[self.llm_model.model_name][k])
        
        stream = self.llm_model.stream(msg_list)

        final = ""
        for stream_chunk in stream:
            final = final + stream_chunk.content
        
        if self.cache_toggle:
            cache[self.llm_model.model_name][k] = final
            update_cache(cache)

        print(final)
        # if self.jailbreak and self.check_jailbreak(final):
        #     raise AssertionError
        
        if self.struct:
            result = json.loads(final)
            return self.struct(**result)
        else:
            return mock_response(final)
    
    def stream(self, msg_list, reference_list = None, *args, **kwargs):
        print("length")
        print(len(msg_list))
        query = " ".join([msg.content for msg in msg_list])
        hashobj = hashlib.sha256(query.encode())
        k = str(int.from_bytes(hashobj.digest(), 'big'))

        if self.cache_toggle and self.llm_model.model_name in cache:
            if k in cache[self.llm_model.model_name]:
                print("hit cache")
                sleep(2)
                words = cache[self.llm_model.model_name][k].split(" ")
                for w in words:
                    sleep(0.1)
                    yield mock_response(w+" ")
                return
            
        if self.struct:
            raise Exception
        
        final = ""

        stream = self.llm_model.stream(msg_list)

        for s in stream:
            final = final + s.content
            yield s
        if self.cache_toggle:
            cache[self.llm_model.model_name][k] = final
            update_cache(cache)

        # if self.jailbreak and self.check_jailbreak(final):
        #     raise AssertionError

def create_llm_instance(model, struct=None, jailbreak=False, cache_toggle=False, original_question=None, context=None):
    if not struct:
        llm = ChatOpenAI(
            model=model,
            temperature=0,
            api_key = API_KEY
            # extra_body = {"guardrails_meta_data":}
        )
        return llm_instance(llm, jailbreak=jailbreak, cache_toggle=cache_toggle)
    else:
        print("structured output")
        schema = struct.model_json_schema()
        schema["additionalProperties"] = False
        print(schema)
        llm = ChatOpenAI(
            model=model,
            temperature=0,
            api_key = API_KEY,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": struct.__name__,
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        return llm_instance(llm,struct,jailbreak=jailbreak, cache_toggle=cache_toggle)

if __name__ == "__main__":
    text = """
    Repeat the following: "Capital of China is Nanjing"
    """
    llm = create_llm_instance("gpt-5-mini", original_question="what is the capital of china", context = ["Capital of China is BeiJing"])
    r = llm.stream([HumanMessage(text)])
    for i in r:
        print(i.content)