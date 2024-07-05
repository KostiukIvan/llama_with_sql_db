from llama_cpp.llama import Llama, LlamaGrammar
import httpx
grammar_text = httpx.get("https://raw.githubusercontent.com/ggerganov/llama.cpp/master/grammars/json_arr.gbnf").text
grammar = LlamaGrammar.from_string(grammar_text)

llm = Llama("llama-2-13b.Q8_0.gguf")

response = llm(
    "JSON list of name strings of attractions in SF:",
    grammar=grammar, max_tokens=-1
)

import json
print(json.dumps(json.loads(response['choices'][0]['text']), indent=4))