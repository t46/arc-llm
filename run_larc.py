# the experiment ! get gpt4 in here
import os
import json
from pathlib import Path
import numpy as np
from src.prompts import nl_and_io_prompt, review

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

def get_llm_response(conversation):
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
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
        few_shot_id = 0
        target_id = few_shot_id + 1
        prompt = nl_and_io_prompt(task, few_shot_id, target_id)
        conv.add_user(prompt)
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