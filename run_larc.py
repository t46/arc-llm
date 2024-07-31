# the experiment ! get gpt4 in here
import os
import json
from pathlib import Path
import numpy as np
from openai import OpenAI
import yaml

from src.prompts import generate_nl_and_io_prompt, generate_review_prompt, generate_test_prediction_prompt
from src.conversation import Coversation
from src.evaluation import eval_score
from src.utils import extract_output

# YAML からイテレーション数などの実験条件の読み込み
with open("config.yaml") as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

# get API key
api_key_path = Path("../../openai_po991_arc.key")
os.environ["OPENAI_API_KEY"] = api_key_path.read_text().strip()
client = OpenAI()

def get_llm_response(conversation):
    response = client.chat.completions.create(
        model=config["model"],
        messages=conversation,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    save_dir = Path('./result')

    with open("data/larc_gpt4.json") as json_file:
        larc_gpt4 = json.load(json_file)
    selected_tasks = larc_gpt4[: config["num_tasks"]]

    for task in selected_tasks:
        conv = Coversation(save_dir=save_dir)
        few_shot_id = 0
        target_id = few_shot_id + 1

        print(f"Round 0")
        prompt = generate_nl_and_io_prompt(task, few_shot_id, target_id)
        conv.add_user(prompt)
        conv.print()
        answer = get_llm_response(conv.history)
        conv.add_assistant(answer)
        conv.print()
        # Error Handling
        try:
            pred_grid = extract_output(answer, "output_grid")
        except:
            print("Error: output_grid tag is not found")
            break
        gt_grid = task["problem"]["train"][target_id]["output"]
        score = eval_score(pred_grid, gt_grid)
        conv.add_score(score)
        if score == 1:
            print("Get the correct answer!")
            print("Break the round loop")
            break

        for round in range(1, config["max_round"] + 1):
            print(f"Round {round}")
            conv.add_user(generate_review_prompt())
            conv.print()
            answer = get_llm_response(conv.history)
            conv.add_assistant(answer)
            conv.print()
            # Error Handling
            try:
                pred_grid = extract_output(answer, "output_grid")
            except:
                print("Error: output_grid tag is not found")
                break
            gt_grid = task["problem"]["train"][target_id]["output"]
            score = eval_score(pred_grid, gt_grid)
            conv.add_score(score)
            if score == 1:
                print("Get the correct answer!")
                print("Break the round loop")
                break
    
        # テストに対する推論
        conv.add_user(generate_test_prediction_prompt(task))
        conv.print()
        answer = get_llm_response(conv.history)
        conv.add_assistant(answer)
        # Error Handling
        try:
            pred_grid = extract_output(answer, "output_grid")
        except:
            print("Error: output_grid tag is not found")
            break
        gt_grid = task["problem"]["test"][0]["output"]
        score = eval_score(pred_grid, gt_grid)
        print(f"Test score: {score}")
        conv.print()

        conv.save_conversation()