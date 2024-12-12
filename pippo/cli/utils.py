import json
import os
import argparse

def load_config(config_path="config.json"):
    """Load configuration from the default config.json file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, 'r') as file:
        return json.load(file)


def parse_arguments():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description="Pippo: A CLI for interacting with various LLMs")
    parser.add_argument("-p", "--prompt", help="Path to the prompt file")
    parser.add_argument("-f", "--files", nargs="*", help="List of files to include in the prompt")
    parser.add_argument("-o", "--output", help="File to save the AI's response")
    parser.add_argument("-q", "--query", help="Direct query to send to the AI model")

    return parser.parse_args()


def read_files(files):
    """Read content from multiple files and return formatted string."""
    missing_files = [file for file in files if not os.path.exists(file)]
    if missing_files:
        raise FileNotFoundError(f"The following files were not found: {', '.join(missing_files)}")

    file_contents = []
    for file in files:
        try:
            with open(file, 'r') as f:
                content = f.read()
                file_contents.append(f"```{file}\n{content}\n```")
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    
    return "\n".join(file_contents)
