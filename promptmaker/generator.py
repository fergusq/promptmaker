import re
from typing import NamedTuple


class Generator:
	def generate(self, prompt: str, params: "GenerationParams") -> str:
		...
	
	def score(self, prompt: str) -> float:
		...


class GenerationParams(NamedTuple):
	max_len: int = 250
	stop: list[re.Pattern] = []
	temperature: float = 0.7