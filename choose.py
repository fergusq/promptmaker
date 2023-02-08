from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Button
from textual.containers import Container

class Option(Static):
	def __init__(self, title: str, desc: str):
		super().__init__()
		self.title = title
		self.desc = desc

	def compose(self) -> ComposeResult:
		yield Button(self.title, id="choose")
		yield Static(id="pad")
		yield Static(self.desc, id="desc")

class Chooser(App):

	BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

	DEFAULT_CSS = """
	Option {
		layout: horizontal;
		background: $boost;
		height: 5;
		margin: 1;
		min-width: 50;
		padding: 1;
	}

	Button {
		dock: left;
	}

	#pad {
		width: 1;
	}
	"""

	def compose(self) -> ComposeResult:
		"""Create child widgets for the app."""
		yield Header()
		yield Footer()
		yield Container(
			Option("Eroottinen runo", "Koeta saada tekoäly tuottamaan eroottinen runo ja koeta kiertää sensurointimalli."),
			Option("Eroottinen tarina", "Koeta saada tekoäly tuottamaan eroottinen tarina ja koeta kiertää sensurointimalli."),
		)

	def action_toggle_dark(self) -> None:
		self.dark = not self.dark
	
	def on_button_pressed(self, event: Button.Pressed) -> None:
		self.exit()
		print(event.button.label)


if __name__ == "__main__":
	app = Chooser()
	app.run()