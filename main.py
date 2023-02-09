import re
import argparse
from pathlib import Path
from typing import Any, Callable

from promptmaker.reader import Reader
from promptmaker.prompt import GeneratorState
from promptmaker.generator import GenerationParams, Generator
from promptmaker.backends.transformers import TransformersGenerator
from promptmaker.backends.openai import OpenAIGenerator

import openai
import rich, rich.console

reader = Reader()
reader.param_confs["word"] = GenerationParams(
	stop=[re.compile(r"\W")],
	max_len=10,
)
reader.param_confs["sentence"] = GenerationParams(
	stop=[re.compile(r"[.!?]")],
	max_len=250,
)
reader.param_confs["line"] = GenerationParams(
	stop=[re.compile("\n")],
	max_len=500,
)
reader.param_confs["paragraph"] = GenerationParams(
	stop=[re.compile("\n\n")],
	max_len=1000,
)
reader.param_confs["text"] = GenerationParams(
	max_len=1000,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="text-davinci-003") # "facebook/xglm-1.7B", "bigscience/bloom-petals"
parser.add_argument("--type", choices=["openai", "transformers", "petals"], default="openai")
parser.add_argument("--var", nargs=2, action="append")
parser.add_argument("--loop", action="store_true")
parser.add_argument("script", type=Path, nargs="?")
args = parser.parse_args()

if args.type == "transformers":
	generator = TransformersGenerator(args.model)
elif args.type == "petals":
	generator = TransformersGenerator(args.model, petals=True)
elif args.type == "openai":
	generator = OpenAIGenerator(args.model)
else:
	assert False

import readline

if args.script:

	console = rich.console.Console()

	class StatusGenerator(Generator):
		def __init__(self, generator: Generator):
			self.generator = generator
		
		def generate(self, *args, **kwargs):
			with console.status(state.globals["status_text"], spinner=state.globals["status_spinner"]):
				return self.generator.generate(*args, **kwargs)

	def safe_input(prompt: str = "", print_report: Callable[[str | None, float], None] | None = None) -> str:
		while True:
			text = input(prompt)
			if not check_string(text, print_report):
				break
		
		return text
	
	def check_string(text: str, print_report: Callable[[str | None, float], None] | None = None) -> bool:
		with console.status(state.globals["status_text"], spinner=state.globals["status_spinner"]):
			response: Any = openai.Moderation.create(text)
		
		#print(response["results"][0]["category_scores"])
		
		category_scores = response["results"][0]["category_scores"]
		argmax = None
		m = 0
		for category, value in category_scores.items():
			if value > m:
				argmax = category
				m = value
		
		if m >= state.globals["threshold"]:
		
			if print_report:
				print_report(argmax, m)
			
			return True
		
		return False

	while True:
		template_text = args.script.read_text()
		template = reader.read(template_text)
		state = GeneratorState(StatusGenerator(generator))
		#for var, val in args.var:
		#	state.vars[var] = val
		state.globals["threshold"] = 0.5
		state.globals["safe_input"] = safe_input
		state.globals["check_string"] = check_string
		state.globals["status_text"] = "Tekoäly miettii..."
		state.globals["status_spinner"] = "dots"
		state.globals["console"] = console
		template.exec(state)
	
		if not args.loop:
			break

else:

	while True:
		try:
			template_text = input("> ").strip()
		except EOFError:
			break

		template = reader.read(template_text)
		state = template.apply(generator)
		print(state.prompt, state.vars)

	# My name is <word NAME!>. I think {NAME} is a cool name, because <sentence REASON>.
