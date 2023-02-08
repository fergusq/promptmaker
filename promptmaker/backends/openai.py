import os
from typing import Any

import openai
from transformers import GPT2TokenizerFast

from ..generator import Generator, GenerationParams

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the GPT-2 tokenizer used for GPT-3
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Load the tokens which do not begin with a space
no_space_tokens = []
for token, id in tokenizer.vocab.items(): # type: ignore
	if not token.startswith("Ġ"):
		no_space_tokens.append(id)

# logprobs table for prventing generation of all tokens which do not contain space
only_space_logit_bias = {str(token): -100 for token in no_space_tokens}
#only_space_logit_bias = {"50256": -100}

class OpenAIGenerator(Generator):
	def __init__(self, model_name: str):
		self.model_name = model_name
	
	def generate(self, prompt: str, params: "GenerationParams") -> str:
		if prompt.endswith(" "):
			space = True
			prompt = prompt.rstrip(" ")
		
		else:
			space = False
		
		if space:
			# First, generate the first token using logprobs
			generated_text = ""
			logit_bias = {"50256": -100}
			while not generated_text.startswith(" "):
				response: Any = openai.Completion.create(
					model=self.model_name,
					prompt=prompt,
					temperature=params.temperature,
					max_tokens=1,
					n=20,
					logit_bias=logit_bias,
				)
				for choice in response["choices"]:
					generated_text = choice["text"]
					if generated_text.startswith(" "):
						break
				
					else:
						logit_bias[str(tokenizer.encode(generated_text))] = -100
			
			if params.max_len > 1:
				# Generate rest of the tokens
				response: Any = openai.Completion.create(
					model=self.model_name,
					prompt=prompt + generated_text,
					temperature=params.temperature,
					max_tokens=params.max_len - 1,
				)
				generated_text += response["choices"][0]["text"]
		else:
			# Normal generation
			response: Any = openai.Completion.create(
				model=self.model_name,
				prompt=prompt,
				temperature=params.temperature,
				max_tokens=params.max_len,
			)

			generated_text = response["choices"][0]["text"]

		if space:
			generated_text = generated_text.lstrip(" ")
		
		for stop_sequence in params.stop:
			if m := stop_sequence.search(generated_text):
				generated_text = generated_text[:m.span(0)[0]]
				break
		
		return generated_text