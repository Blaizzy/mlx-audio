"""Main CLI entry point for mlx-audio.

This module provides a unified command-line interface that dispatches to
different subcommands like 'generate' and 'server'.
"""

import sys


def print_help():
    """Print the help message for the main CLI."""
    print("Usage: mlx_audio <command> [options]")
    print("\nAvailable commands:")
    print("  generate    Generate audio from text using TTS")
    print("  server      Start the MLX Audio API server")
    print("\nFor command-specific help, run: mlx_audio <command> --help")


def main():
    """Main CLI entry point that dispatches to subcommands."""
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Remove the command from argv so the subcommand's argparse works correctly
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == "generate":
        from mlx_audio.tts.generate import main as generate_main
        generate_main()
    elif command == "server":
        from mlx_audio.server import main as server_main
        server_main()
    else:
        print(f"Error: Unknown command '{command}'\n")
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

