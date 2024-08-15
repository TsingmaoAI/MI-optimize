from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto
model_path = "/models/qwen2-7b-instruct"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.to(device)

# prompt = "Give me a short introduction to large language model."
# messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
messages = [{"role": "user", "content": "你好啊，能帮我讲一下什么是大模型吗？"}]
messages = [{'role': 'user', 'content': '使用位填充方法，以01111110为位首flag，数据为011011111111111111110010，求问传送时要添加几个0____\nA. 1\nB. 2\nC. 3\nD. 4\n答案：'}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("==========================================================")
print(response)
print("==========================================================")
