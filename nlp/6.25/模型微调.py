from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "01-ai/Yi-Coder-1.5B-Chat",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "01-ai/Yi-Coder-1.5B-Chat",
    trust_remote_code=True
)