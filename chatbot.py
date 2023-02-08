import re

from promptmaker.reader import Reader
from promptmaker.generator import GenerationParams
from promptmaker.backends.transformers import TransformersGenerator

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

#generator = TransformersGenerator("facebook/xglm-1.7B")
generator = TransformersGenerator("bigscience/bloom-petals", petals=True)

template_text = """{while True}\
Human: {input("Human: ")}
Friendly AI: <line ANSWER>{print("AI:", ANSWER)}
{endwhile}"""

template = reader.read(template_text)
print(repr(template))
state = template.apply(generator)
print(state.prompt, state.vars)