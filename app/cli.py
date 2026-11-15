"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="game-dialogue-assistant", description="Game designers use this tool to generate dialogue snippets for NPCs based on a corpus of game scripts, ensuring character consistency and rapid prototyping.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    # TODO: add your own commands here as you work through the milestones.
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Game Dialogue Assistant is ready. Open README.md for your milestones.")


if __name__ == "__main__":
    main()
