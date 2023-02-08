import re

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList

try:
	from petals import DistributedBloomForCausalLM
except:
	DistributedBloomForCausalLM = None


from ..generator import Generator, GenerationParams


class TokenStoppingCriteria(StoppingCriteria):
	def __init__(self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, prompt_len: int, sequences: list[re.Pattern]):
		self.tokenizer = tokenizer
		self.prompt_len = prompt_len
		self.sequences = sequences
	
	def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
		#print(input_ids[0, self.prompt_len:], end=" ")
		text = self.tokenizer.decode(input_ids[0, self.prompt_len:])
		#print(repr(text))
		for sequence in self.sequences:
			if sequence.search(text):
				return True
		
		return False


class BanTokensAtStartLogitsProcessor(LogitsProcessor):
	def __init__(self, prompt_len: int, tokens: list[int]):
		self.prompt_len = prompt_len
		self.tokens = tokens
	
	def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
		if input_ids.shape[-1] == self.prompt_len + 1:
			scores[:, self.tokens] = 0
		
		return scores


class TransformersGenerator(Generator):
	def __init__(self, model_name: str, petals=False):
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)

		self.petals = petals
		if petals:
			assert DistributedBloomForCausalLM
			self.model = DistributedBloomForCausalLM.from_pretrained(model_name) # type: ignore
		
		else:
			self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

		self.no_space_tokens = []
		for token, id in self.tokenizer.get_vocab().items(): # type: ignore
			if not token.startswith(" ") and not token.startswith("▁") and not token.startswith("Ġ"): # TODO: figure out space character automatically
				self.no_space_tokens.append(id)

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model.to(self.device)
	
	def generate(self, prompt: str, params: GenerationParams):
		if prompt.endswith(" "):
			space = True
			prompt = prompt.rstrip(" ")
		
		else:
			space = False
		
		tokenized = self.tokenizer(prompt, return_tensors="pt")
		input_ids = tokenized["input_ids"].to(self.device) # type: ignore
		
		#print("Generating", repr(prompt), params)
		stopping_criteria = TokenStoppingCriteria(self.tokenizer, input_ids.shape[-1], params.stop)

		logits_processor = None
		if space:
			logits_processor = LogitsProcessorList([BanTokensAtStartLogitsProcessor(input_ids.shape[-1], self.no_space_tokens)])
		
		if self.petals:
			out_ids = self.generate_petals(
				input_ids=input_ids,
				max_length=params.max_len,
				temperature=params.temperature,
				space=space,
				stopping_criteria=stopping_criteria
			)
		else:
			out = self.model.generate(
				input_ids,
				min_length=1,
				max_new_tokens=params.max_len, 
				num_beams=1,
				num_return_sequences=1,
				do_sample=True,
				temperature=params.temperature,
				stopping_criteria=StoppingCriteriaList([stopping_criteria]),
				logits_processor=logits_processor,
				attention_mask=tokenized.get("attention_mask", None),
				pad_token_id=self.tokenizer.eos_token_id,
			)
			out_ids = out[0, input_ids.shape[-1]:]
		#print("OUT", out_ids)

		generated_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
		if space:
			generated_text = generated_text.lstrip(" ")
		
		for stop_sequence in params.stop:
			if m := stop_sequence.search(generated_text):
				generated_text = generated_text[:m.span(0)[0]]
				break
		
		#print("Generated", repr(generated_text))
		return generated_text
	
	def generate_petals(
		self,
		input_ids: torch.LongTensor,
		max_length: int,
		temperature: float,
		space: bool,
		stopping_criteria: StoppingCriteria | None,
	) -> list[int]:
		result = []
		token_ids = input_ids
		with torch.inference_mode():
			with self.model.inference_session(max_length=max_length) as sess:
				while len(result) < max_length:
					embs = self.model.transformer.word_embeddings(token_ids)
					embs = self.model.transformer.word_embeddings_layernorm(embs)

					h = sess.step(embs)
					h_last = self.model.transformer.ln_f(h[:, -1])
					logits = self.model.lm_head(h_last)

					logits = logits.softmax(dim=-1)
					logits.pow_(1/temperature)

					if space and len(result) == 0:
						logits[0, self.no_space_tokens] = 0

					next_token = logits.multinomial(1)[0]
					result.append(int(next_token[0]))
					#print(result)
					t = self.tokenizer.decode(next_token)
					print(repr(t))
					token_ids = next_token.reshape(1, 1)

					if t == "</s>":
						break

					if stopping_criteria and stopping_criteria(torch.concat((input_ids, torch.tensor([result]).to(self.device)), dim=-1), logits): # type: ignore
						break
		
		return result