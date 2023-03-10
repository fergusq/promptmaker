from types import CodeType

from .generator import Generator, GenerationParams


class GeneratorState:
	generator: Generator
	prompt: str
	vars: dict[str, str | list[str]]
	globals = {}

	def __init__(self, generator: Generator):
		self.generator = generator
		self.prompt = ""
		self.vars = {}


class PromptAction:
	def exec(self, state: GeneratorState):
		...


class PromptTemplate(PromptAction):
	actions: list[PromptAction]

	def __init__(self):
		self.actions = []
	
	def apply(self, generator: Generator):
		state = GeneratorState(generator)
		self.exec(state)
		return state
	
	def exec(self, state: GeneratorState):
		for action in self.actions:
			action.exec(state)
	
	def __repr__(self):
		return f"<PromptTemplate actions={self.actions}>"


class AppendPromptAction(PromptAction):
	def __init__(self, text: str):
		self.text = text

	def exec(self, state: GeneratorState):
		if state.prompt == "":
			state.prompt += self.text.lstrip()
		
		else:
			state.prompt += self.text
	
	def __repr__(self):
		return f"<AppendPromptAction text={repr(self.text)}>"


class WhileAction(PromptAction):
	def __init__(self, cond: str, body: PromptTemplate):
		self.cond = cond
		self.body = body
	
	def exec(self, state: GeneratorState):
		while True:
			result = eval(self.cond, state.globals, {"state": state, **state.vars})
			if not result:
				break
			
			self.body.exec(state)
	
	def __repr__(self):
		return f"<WhileAction cond={repr(self.cond)} body={repr(self.body)}>"


class IfAction(PromptAction):
	def __init__(self, cond: str, body: PromptTemplate):
		self.cond = cond
		self.body = body
	
	def exec(self, state: GeneratorState):
		result = eval(self.cond, state.globals, {"state": state, **state.vars})
		if result:
			self.body.exec(state)
	
	def __repr__(self):
		return f"<IfAction cond={repr(self.cond)} body={repr(self.body)}>"


class HiddenAction(PromptAction):
	def __init__(self, body: PromptTemplate):
		self.body = body
	
	def exec(self, state: GeneratorState):
		prompt = state.prompt
		self.body.exec(state)
		state.prompt = prompt
	
	def __repr__(self):
		return f"<HiddenAction body={repr(self.body)}>"


class EvalAction(PromptAction):
	def __init__(self, code: str | CodeType):
		self.code = code
	
	def exec(self, state: GeneratorState):
		result = eval(self.code, state.globals, {"state": state, **state.vars})
		if result is not None:
			state.prompt += str(result)
	
	def __repr__(self):
		return f"<EvalAction code={repr(self.code)}>"


class ReadVariableAction(PromptAction):
	def __init__(self, var: str, params: GenerationParams, append = False, strip = False):
		self.var = var
		self.params = params
		self.append = append
		self.strip = strip
	
	def exec(self, state: GeneratorState):
		text = state.generator.generate(state.prompt, self.params)
		state.prompt += text

		if self.strip:
			text = text.strip()

		if self.append:
			l = state.vars.get(self.var, [])
			assert isinstance(l, list)
			l.append(text)
		else:
			state.vars[self.var] = text
	
	def __repr__(self):
		return f"<ReadVariableAction var={repr(self.var)}>"

class ReadAlternativeToVariableAction(PromptAction):
	def __init__(self, alternatives: list[str], var: str, params: GenerationParams):
		self.alternatives = alternatives
		self.var = var
		self.params = params
	
	def exec(self, state: GeneratorState):
		if self.params.temperature != 0:
			raise NotImplementedError
		
		results: list[tuple[float, str, str]] = []
		scores = state.generator.scores([state.prompt + alternative for alternative in self.alternatives])
		for alternative, score in zip(self.alternatives, scores):
			results.append((score, state.prompt + alternative, alternative))
		
		state.prompt, state.vars[self.var] = sorted(results, key=lambda s: -s[0])[0][1:]
	
	def __repr__(self):
		return f"<ReadAlternativeToVariableAction var={repr(self.var)} alternatives={repr(self.alternatives)}>"