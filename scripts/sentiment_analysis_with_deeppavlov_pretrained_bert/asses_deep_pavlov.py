from deeppavlov import build_model
import json
from tqdm import tqdm

with open("/home/ssorokin/BeWell/notebooks/BERT_experiments/vk_texts_3_months.json", "r") as write_file:
    texts = json.load(write_file)
    
model = build_model("rusentiment_bert", download=True)

for vk_id, messeges in tqdm(list(texts.items())):
    asses = model([str(i) for i in messeges])
    
    with open(f"./asses/{vk_id}.json", "w") as write_file:
        json.dump({vk_id:(messeges, asses)}, write_file, indent=4)
    
