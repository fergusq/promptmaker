import re
from typing import NamedTuple

from .generator import GenerationParams
from .prompt import PromptTemplate, AppendPromptAction, EvalAction, ReadVariableAction, ReadAlternativeToVariableAction, WhileAction, IfAction, HiddenAction


class Token(NamedTuple):
	start: int
	end: int
	text: str


class ParenToken(Token):
	def get_paren(self):
		return self.text[0]

	def get_content(self):
		return self.text[1:-1].strip()
	
	def get_head(self):
		content = self.get_content()
		if " " not in content:
			return content
		
		else:
			return content[:content.index(" ")]
	
	def get_tail(self):
		content = self.get_content()
		if " " not in content:
			return ""
		
		else:
			return content[content.index(" ")+1:].strip()


def lex(text: str) -> list[Token]:
	ans = []
	
	i = 0
	while i < len(text):
		pos = i
		if text[i] not in "<{@":
			acc = ""
			while i < len(text) and text[i] not in "<{@":
				char = text[i]; i += 1
				if char == "\\" and i < len(text):
					next = text[i]; i += 1
					if next.isspace():
						continue
					
					else:
						acc += next
				
				else:
					acc += char
			
			if acc:
				ans.append(Token(pos, i, acc))
		
		elif text[i] in "<{@" and i < len(text):
			paren_open, paren_close = {
				"<": ("<", ">"),
				"{": ("{", "}"),
				"@": ("@", "\n"),
			}[text[i]]
			i += 1
			depth = 0
			acc = ""
			while i < len(text):
				char = text[i]; i += 1
				if char == paren_open:
					depth += 1
				
				if char == paren_close:
					depth -= 1
				
				if depth == -1:
					break
				
				else:
					acc += char

			if acc:
				ans.append(ParenToken(pos, i, paren_open+acc+paren_close))
	
	return ans


class ParseError(Exception):
	def __init__(self, token: Token | None, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.token = token


def expect_head(tokens: list[Token], head: str) -> list[Token]:
	if not tokens:
		raise ParseError(None, "unexpected EOF, expected {" + head + "}")
	
	if not isinstance(tokens[0], ParenToken) or tokens[0].get_head() != head:
		raise ParseError(tokens[0], "expected {" + head + "}")
	
	return tokens[1:]


class Reader:
	param_confs: dict[str, GenerationParams]
	def __init__(self):
		self.param_confs = {}
	
	def read(self, text: str) -> "PromptTemplate":
		tokens = lex(text)
		tokens, template = self.read_script(tokens)
		if len(tokens) != 0:
			raise ParseError(tokens[0], "expected EOF")
		
		return template
	
	def read_script(self, tokens: list[Token]):
		template = PromptTemplate()
		while tokens:
			token = tokens[0]
			if isinstance(token, ParenToken):
				if token.get_paren() == "{":
					if token.get_head() in {"endwhile", "endif", "endhidden"}:
						break

					tokens, action = self.read_expr(tokens)
					template.actions.append(action)
				
				elif token.get_paren() == "<":
					tokens, action = self.read_variable(tokens)
					template.actions.append(action)

				else:
					assert False, token
			
			else:
				template.actions.append(AppendPromptAction(tokens[0].text))
				tokens = tokens[1:]
		
		return tokens, template
	
	def read_expr(self, tokens: list[Token]):
		expr_token = tokens[0]
		assert isinstance(expr_token, ParenToken)
		
		if expr_token.get_head() == "while":
			cond = expr_token.get_tail()
			tokens, body = self.read_script(tokens[1:])
			tokens = expect_head(tokens, "endwhile")
			return tokens, WhileAction(cond, body)
		
		elif expr_token.get_head() == "if":
			cond = expr_token.get_tail()
			tokens, body = self.read_script(tokens[1:])
			tokens = expect_head(tokens, "endif")
			return tokens, IfAction(cond, body)
		
		elif expr_token.get_head() == "hidden":
			assert expr_token.get_tail() == ""
			tokens, body = self.read_script(tokens[1:])
			tokens = expect_head(tokens, "endhidden")
			return tokens, HiddenAction(body)
		
		elif expr_token.text[1:].startswith("\n"):
			return tokens[1:], EvalAction(compile(expr_token.get_content(), "<inline>", "exec"))
		
		else:
			return tokens[1:], EvalAction(expr_token.get_content())
	
	def read_variable(self, tokens: list[Token]):
		var_token = tokens[0]
		assert isinstance(var_token, ParenToken)

		var_code = var_token.get_content()

		if m := re.fullmatch(r"\(([^)]+\|[^)]+)\) *([^?]*)(\??)", var_code):
			alternatives = m.group(1).split("|")
			variable = m.group(2)
			greedy = m.group(3) == "?"
			params = GenerationParams()
			if greedy:
				params = params._replace(temperature=0)
			return tokens[1:], ReadAlternativeToVariableAction(alternatives, variable, params)

		fields = var_code.split(":")
		var_code = fields[0]
		
		append = False
		strip = False
		greedy = False
		while var_code[-1:] in {"+", "!", "?"}:
			if var_code[-1] == "+":
				append = True
			
			elif var_code[-1] == "!":
				strip = True
			
			elif var_code[-1] == "?":
				greedy = True
			
			var_code = var_code[:-1]
		
		params = GenerationParams()
		if greedy:
			params = params._replace(temperature=0)
		
		if " " in var_code:
			words = var_code.split(" ")
			var_code = words[-1]
			for param_conf_name in words[:-1]:
				params = params._replace(**self.param_confs[param_conf_name]._asdict())
		
		for field in fields[1:]:
			assert field.count("=") == 1
			[name, value] = field.split("=")
			name = name.strip()
			value = value.strip()
			assert name and value
			assert name in GenerationParams._fields
			field_type = type(GenerationParams._field_defaults[name])
			if field_type == int or field_type == float or field_type == bool:
				value = field_type(value)
			
			else:
				assert field_type == str
			
			params = params._replace(**{name: value})
		
		return tokens[1:], ReadVariableAction(var_code, params, append=append, strip=strip)



