# the experiment ! get gpt4 in here
import os
import json
from pathlib import Path

import numpy as np
from openai import OpenAI

# get API key
api_key_path = Path("../../openai_po991_arc.key")
os.environ["OPENAI_API_KEY"] = api_key_path.read_text().strip()
client = OpenAI()

def eval_score(pred_grid, gt_grid, show=True):
    pred_grid = np.array(eval(str(pred_grid)))
    gt_grid = np.array(eval(str(gt_grid)))
    pred_shape = pred_grid.shape
    gt_shape = gt_grid.shape

    if pred_shape != gt_shape:
        score = 0.0
    else:
        score = (pred_grid == gt_grid).sum() / (pred_shape[0] * pred_shape[1])

    if show:
        print("\033[93m" + f"score: {score}" + "\033[0m")

    return score

preamble = """
lets play a game where you are transforming an input grid of numbers into an output grid of numbers

the numbers represent different colors:
0 = black
1 = blue
2 = red
3 = green
4 = yellow
5 = gray
6 = magenta
7 = orange
8 = cyan
9 = brown

"""

def nl_and_io_prompt(task):
    instruction = "here is the instruction of how to transform the grid: \n"
    instruction += (
        task["description"]["description_input"]
        + task["description"]["description_output_grid_size"]
        + task["description"]["description_output"]
    )

    train_input = task["problem"]["train"][0]["input"]
    train_output = task["problem"]["train"][0]["output"]
    input_output_example = (
        "\n\nhere is an example of an input grid and its corresponding output grid:\n"
    )
    input_output_example += (
        "example input grid:\n"
        + str(train_input)
        + "\nexample output grid:\n"
        + str(train_output)
        + "\n\n"
    )

    input_grid = task["problem"]["train"][1]["input"]

    prompt = (
        preamble
        + instruction
        + input_output_example
        + "\n\nThe input grid is:\n"
        + str(input_grid)
        + "\n\nWhat is the output grid?"
        + "\n\nOutput gird surrounded by <output_grid> and <output_grid>"
    )
    return prompt


def review(grid):
    prompt = f"""Your output was incorrect.

Please clearly identify the differences between the correct answer and your output.
Specifically, highlight which part of the given task description was not accurately executed, resulting in this discrepancy.
Then, provide the correct answer based on the correct interpretation of the task.
Output gird surrounded by <output_grid> and <output_grid>
"""
    return prompt


def get_llm_response(conversation):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation,
    )
    return response.choices[0].message.content

class Coversation:
    def __init__(self, save_dir):
        self.conversation = [
            {"role": "system", "content": "You are an autonomous task solver."}
        ]
        self.roles = ['system']
        self.save_dir = save_dir

    def add_user(self, txt):
        self.conversation.append({"role": "user", "content": txt})
        self.roles.append("user")

    def add_assistant(self, txt):
        self.conversation.append({"role": "assistant", "content": txt})
        self.roles.append("assistant")

    @property
    def history(self):
        return self.conversation
    
    def _print(self, conv):
        print("=" * 20)
        print(f"[{conv['role']}]")
        print(conv['role'])
        print("=" * 20)
        print(conv["content"])
        print('-*' * 40)

    def print(self):
        conv = self.conversation[-1]
        self._print(conv)

    def print_history(self):
        for conv in self.conversation:
            self._print(conv)
    
    def save_conversation(self):
        files = save_dir.glob("*.json")
        path = self.save_dir / f'{str(len(list(files))).zfill(3)}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.conversation, f, indent=4)

if __name__ == "__main__":
    save_dir = Path('./result')

    with open("data/larc_gpt4.json") as json_file:
        larc_gpt4 = json.load(json_file)
    selected_tasks = larc_gpt4[:1]

    for task in selected_tasks:
        conv = Coversation(save_dir=save_dir)
        conv.add_user(nl_and_io_prompt(task))
        conv.print()
        conv.add_assistant(get_llm_response(conv.history))
        conv.print()

        scores = []
        for round in range(2):
            print(f"Round {round}")
            conv.add_user(review(task["problem"]["train"][1]["output"]))
            conv.print()
            answer = get_llm_response(conv.history)
            conv.add_assistant(answer)
            conv.print()

            pred_grid = answer.split("<output_grid>")[1].split("</output_grid>")[0]
            gt_grid = task["problem"]["train"][1]["output"]

            score = eval_score(pred_grid, gt_grid)
            scores.append(score)
            if score == 1:
                print("Get the correct answer!")
                print("Break the round loop")
                break
        
        conv.save_conversation()