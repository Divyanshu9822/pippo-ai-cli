from pippo.cli.utils import parse_arguments, load_config, read_files
from pippo.utils.factory import LlmFactory
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    args = parse_arguments()

    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    model_name = config["llm"]["provider"]
    try:
        llm_config = config["llm"].get("config", {})
        llm_instance = LlmFactory.create(model_name, llm_config)
    except ValueError as e:
        print(f"Error: {e}")
        return

    prompt_content = ""
    if args.prompt:
        try:
            with open(args.prompt, 'r') as prompt_file:
                prompt_content = prompt_file.read()
        except Exception as e:
            print(f"Error reading prompt file: {e}")
            return

    if args.query:
        if args.prompt:
            print("Error: You cannot provide both a prompt file and a query at the same time.")
            return
        prompt_content = args.query

    if not prompt_content:
        print("Error: You must provide either a prompt file (-p) or a query (-q).")
        return

    files_content = ""
    if args.files:
        files_content = read_files(args.files)

    combined_prompt = f"{files_content}\n{prompt_content}"

    messages = [{"role": "user", "content": combined_prompt}]

    try:
        response = llm_instance.generate_response(messages)
        print("AI Response:", response)

        if args.output:
            with open(args.output, "w") as output_file:
                output_file.write(response)
            print(f"Response saved to {args.output}")
    except Exception as e:
        print(f"Error during LLM generation: {e}")
