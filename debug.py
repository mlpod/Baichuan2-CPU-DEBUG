from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("model", trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("model", trust_remote_code=True)
print(model)

messages = []
messages.append({"role": "user", "content": "测试一下能不能运行"})
response = model.chat(tokenizer, messages)

print(response)


