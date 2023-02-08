import torch
from transformers import BloomTokenizerFast 
from petals import DistributedBloomForCausalLM

MODEL_NAME = "bigscience/bloom-petals"
tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME)
model = model.cuda()

no_space_tokens = []
for token, id in tokenizer.vocab.items(): # type: ignore
	if not token.startswith(" ") and not token.startswith("▁") and not token.startswith("Ġ"): # TODO: figure out space character automatically
		no_space_tokens.append(id)

text = "My name is"
token_ids = tokenizer.encode(text, return_tensors="pt").cuda()
max_length = 100
result = []
with torch.inference_mode():
	with model.inference_session(max_length=max_length) as sess:
		while len(result) < max_length:
			embs = model.transformer.word_embeddings(token_ids)
			embs = model.transformer.word_embeddings_layernorm(embs)

			h = sess.step(embs)
			h_last = model.transformer.ln_f(h[:, -1])
			logits = model.lm_head(h_last)

			logits = logits.softmax(dim=-1)
			logits.pow_(1/0.7)

			if len(result) == 0:
				logits[0, no_space_tokens] = 0

			next_token = logits.multinomial(1)[0]
			text += tokenizer.decode(next_token)
			result.append(next_token)
			token_ids = next_token.reshape(1, 1)
			print(text, result)