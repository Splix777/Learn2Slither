"""Command line interface for training and evaluating the snake AI."""

from typing import Optional

import typer
from rich.prompt import Prompt

from src.config.settings import config
from src.game.environment import Environment
from src.game.snake import Snake
from src.ai.agent import Agent
from src.ai.utils.plotter import Plotter
from src.ai.trainer import ReinforcementLearner
from src.ui.pygame.pygame_gui import PygameGUI

app = typer.Typer()


@app.command()
def train(
    sessions: Optional[int] = 100,
    save: Optional[str] = None,
    visualize: bool = False,
    fast: bool = False,
    plot: bool = False,
    early_stop: bool = False,
):
    config.nn.epochs = sessions or config.nn.epochs
    gui = None
    plotter = None
    if visualize:
        gui = PygameGUI(
            config,
            (
                config.map.board_size.width * config.textures.texture_size,
                config.map.board_size.height * config.textures.texture_size,
            ),
        )
    if plot:
        plotter = Plotter()
    snake = Snake(0, brain=Agent(config), config=config)
    env = Environment(config, [snake])
    trainer = ReinforcementLearner(
        config=config,
        env=env,
        early_stop=early_stop,
        plotter=plotter,
        gui=gui,
        save_path=save,
    )

    try:
        trainer.train(fast)

    except KeyboardInterrupt:
        typer.echo("Training interrupted by user.")
    except Exception as e:
        typer.echo(f"An error occurred: {e}")


@app.command()
def evaluate(
    episodes: int = 5,
    load: Optional[str] = None,
    visualize: bool = False,
    fast: bool = False,
    step_by_step: bool = False,
    map_size: Optional[int] = 10,
):
    gui = None
    plotter = None
    while map_size not in [10, 20, 30]:
        map_size = int(
            Prompt.ask("Choose a map size", choices=["10", "20", "30"])
        )
    config.map.board_size.width = map_size
    config.map.board_size.height = map_size
    if visualize:
        gui = PygameGUI(
            config,
            (
                config.map.board_size.width * config.textures.texture_size,
                config.map.board_size.height * config.textures.texture_size,
            ),
        )
    if step_by_step:
        typer.echo(
            "Press Enter to step through the evaluation or exit to quit."
        )
    snake = Snake(0, brain=Agent(config, load), config=config)
    env = Environment(config, [snake])
    trainer = ReinforcementLearner(
        config=config,
        env=env,
        plotter=plotter,
        gui=gui,
    )

    try:
        trainer.evaluate(episodes, fast, step_by_step)

    except KeyboardInterrupt:
        typer.echo("Evaluation interrupted by user.")
    except Exception as e:
        typer.echo(f"An error occurred: {e}")


@app.command()
def game() -> None:
    try:
        gui = PygameGUI(config)
        gui.run()

    except KeyboardInterrupt:
        typer.echo("Game interrupted by user.")
    except Exception as e:
        typer.echo(f"An error occurred: {e}")


if __name__ == "__main__":
    app()
