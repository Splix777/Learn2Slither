"""Command line interface for training and evaluating the snake AI.

This module provides a CLI interface for:
- Training the snake AI
- Evaluating trained models
- Running the game in human-playable mode
- Running a battle royale mode with multiple AI snakes
"""

from typing import Optional
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich import print as rprint

from src.config.settings import config
from src.game.environment import Environment
from src.game.snake import Snake
from src.ai.agent import Agent
from src.ai.utils.plotter import Plotter
from src.ai.trainer import ReinforcementLearner
from src.ui.pygame.pygame_gui import PygameGUI

app = typer.Typer(help="Snake AI Training and Evaluation Tool")
console = Console()

def setup_gui(board_width: int, board_height: int) -> PygameGUI:
    """Initialize and return a pygame GUI instance."""
    return PygameGUI(
        config,
        (
            board_width * config.textures.texture_size,
            board_height * config.textures.texture_size,
        ),
    )

def create_trainer(
    config,
    snake: Snake,
    *,
    early_stop: bool = False,
    plotter: Optional[Plotter] = None,
    gui: Optional[PygameGUI] = None,
    save_path: Optional[str] = None,
) -> ReinforcementLearner:
    """Create and return a trainer instance with the specified configuration."""
    env = Environment(config, [snake])
    return ReinforcementLearner(
        config=config,
        env=env,
        early_stop=early_stop,
        plotter=plotter,
        gui=gui,
        save_path=save_path,
    )

@app.command()
def train(
    sessions: Optional[int] = typer.Option(
        100,
        "--sessions", "-s",
        help="Number of training sessions to run"
    ),
    save: Optional[Path] = typer.Option(
        None,
        "--save",
        help="Path to save the trained model",
        dir_okay=True,
    ),
    visualize: bool = typer.Option(
        False,
        "--visualize", "-v",
        help="Enable visualization during training"
    ),
    fast: bool = typer.Option(
        False,
        "--fast", "-f",
        help="Enable fast mode training"
    ),
    plot: bool = typer.Option(
        False,
        "--plot", "-p",
        help="Enable performance plotting"
    ),
    early_stop: bool = typer.Option(
        False,
        "--early-stop",
        help="Enable early stopping"
    ),
):
    """Train a new snake AI model."""
    try:
        config.nn.epochs = sessions or config.nn.epochs
        gui = setup_gui(config.map.board_size.width, config.map.board_size.height) if visualize else None
        plotter = Plotter() if plot else None
        
        snake = Snake(0, brain=Agent(config), config=config)
        trainer = create_trainer(
            config,
            snake,
            early_stop=early_stop,
            plotter=plotter,
            gui=gui,
            save_path=str(save) if save else None
        )

        with console.status("[bold green]Training in progress..."):
            trainer.train(fast)
        
        rprint("[bold green]Training completed successfully!")

    except KeyboardInterrupt:
        rprint("[yellow]Training interrupted by user.")
    except Exception as e:
        rprint(f"[red]An error occurred: {e}")
        raise typer.Exit(code=1)

@app.command()
def evaluate(
    episodes: int = typer.Option(
        5,
        "--episodes", "-e",
        help="Number of evaluation episodes"
    ),
    load: Optional[Path] = typer.Option(
        None,
        "--load", "-l",
        help="Path to load a trained model",
        exists=True,
    ),
    visualize: bool = typer.Option(
        False,
        "--visualize", "-v",
        help="Enable visualization"
    ),
    fast: bool = typer.Option(
        False,
        "--fast", "-f",
        help="Enable fast mode"
    ),
    step_by_step: bool = typer.Option(
        False,
        "--step",
        help="Enable step-by-step evaluation"
    ),
    map_size: int = typer.Option(
        10,
        "--size",
        help="Map size (10, 20, or 30)"
    ),
):
    """Evaluate a trained snake AI model."""
    try:
        if map_size not in {10, 20, 30}:
            map_size = int(Prompt.ask(
                "Choose a map size",
                choices=["10", "20", "30"],
                default="10"
            ))

        config.map.board_size.width = config.map.board_size.height = map_size
        gui = setup_gui(map_size, map_size) if visualize else None

        if step_by_step:
            rprint("[yellow]Press Enter to step through the evaluation or Ctrl+C to quit.")

        snake = Snake(0, brain=Agent(config, str(load) if load else None), config=config)
        trainer = create_trainer(config, snake, gui=gui)

        with console.status("[bold green]Evaluating model..."):
            trainer.evaluate(episodes, fast, step_by_step)
        
        rprint("[bold green]Evaluation completed successfully!")

    except KeyboardInterrupt:
        rprint("[yellow]Evaluation interrupted by user.")
    except Exception as e:
        rprint(f"[red]An error occurred: {e}")
        raise typer.Exit(code=1)

@app.command()
def game():
    """Run the snake game in human-playable mode."""
    try:
        gui = setup_gui(config.map.board_size.width, config.map.board_size.height)
        with console.status("[bold green]Starting game..."):
            gui.run()
        
    except KeyboardInterrupt:
        rprint("[yellow]Game interrupted by user.")
    except Exception as e:
        rprint(f"[red]An error occurred: {e}")
        raise typer.Exit(code=1)

@app.command()
def battle_royale(
    episodes: int = typer.Option(
        1,
        "--episodes", "-e",
        help="Number of battle episodes"
    ),
    fast: bool = typer.Option(
        False,
        "--fast", "-f",
        help="Enable fast mode"
    ),
    step_by_step: bool = typer.Option(
        False,
        "--step",
        help="Enable step-by-step mode"
    ),
):
    """Run a battle royale mode with multiple AI snakes."""
    try:
        config.map.board_size.width = config.map.board_size.height = 30
        config.map.green_apples = 10
        config.map.red_apples = 5

        gui = setup_gui(30, 30)
        mono_brain = Agent(config, config.snake.difficulty.expert)
        snakes = [Snake(i, brain=mono_brain, config=config) for i in range(10)]
        
        env = Environment(config, snakes)
        trainer = ReinforcementLearner(config=config, env=env, gui=gui)

        with console.status("[bold green]Running battle royale..."):
            trainer.evaluate(
                test_episodes=episodes,
                fast=fast,
                step_by_step=step_by_step,
            )
        
        rprint("[bold green]Battle royale completed successfully!")

    except KeyboardInterrupt:
        rprint("[yellow]Battle royale interrupted by user.")
    except Exception as e:
        rprint(f"[red]An error occurred: {e}")
        raise typer.Exit(code=1)

def main():
    """Entry point for the CLI application."""
    try:
        app()
    except Exception as e:
        console.print_exception()
        raise typer.Exit(code=1)

if __name__ == "__main__":
    main()
