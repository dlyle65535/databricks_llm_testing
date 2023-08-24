# Databricks notebook source
# MAGIC %pip install -U langchain==0.0.251 bitsandbytes==0.40.1 accelerate 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from flask import Flask, request, jsonify
import torch
import transformers
from transformers import pipeline, AutoTokenizer, StoppingCriteria, BitsAndBytesConfig, AutoModelForCausalLM

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_id = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
model=AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="cuda:0", load_in_4bit=True)
falcon7b = pipeline(
    torch_dtype=torch.bfloat16,
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",
    quantization_config= quantization_config,
)
device = falcon7b.device

class CheckStop(StoppingCriteria):
    def __init__(self, stop=None):
        super().__init__()
        self.stop = stop or []
        self.matched = ""
        self.stop_ids = [tokenizer.encode(s, return_tensors='pt').to(device) for s in self.stop]
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for i, s in enumerate(self.stop_ids):
            if torch.all((s == input_ids[0][-s.shape[1]:])).item():
                self.matched = self.stop[i]
                return True
        return False

def llm(prompt, stop=None, **kwargs):
  check_stop = CheckStop(stop)
  result = falcon7b(prompt, stopping_criteria=[check_stop], **kwargs)
  return result[0]["generated_text"].rstrip(check_stop.matched)

app = Flask("falcon7b")

@app.route('/', methods=['POST'])
def serve_llm():
  resp = llm(**request.json)
  return jsonify(resp)

app.run(host="0.0.0.0", port="7777")

# COMMAND ----------

#databricks/dolly-v2-12b
from flask import Flask, request, jsonify
import torch
from transformers import pipeline, AutoTokenizer, StoppingCriteria

model = "databricks/dolly-v2-3b"
tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
dolly = pipeline(model=model, tokenizer=tokenizer, device_map="cuda:0", trust_remote_code=True)
device = dolly.device

class CheckStop(StoppingCriteria):
    def __init__(self, stop=None):
        super().__init__()
        self.stop = stop or []
        self.matched = ""
        self.stop_ids = [tokenizer.encode(s, return_tensors='pt').to(device) for s in self.stop]
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for i, s in enumerate(self.stop_ids):
            if torch.all((s == input_ids[0][-s.shape[1]:])).item():
                self.matched = self.stop[i]
                return True
        return False

def llm(prompt, stop=None, **kwargs):
  check_stop = CheckStop(stop)
  result = dolly(prompt, stopping_criteria=[check_stop], **kwargs)
  return result[0]["generated_text"].rstrip(check_stop.matched)

app = Flask("dolly")

@app.route('/', methods=['POST'])
def serve_llm():
  resp = llm(**request.json)
  return jsonify(resp)

app.run(host="0.0.0.0", port="7777")

# COMMAND ----------


