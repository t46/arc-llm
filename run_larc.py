# the experiment ! get gpt4 in here
import os
import json
from pathlib import Path
import numpy as np
from openai import OpenAI

from src.prompts import generate_nl_and_io_prompt, generate_review_prompt, generate_test_prediction_prompt
from src.conversation import Coversation
from src.evaluation import eval_score
from src.utils import extract_output

# get API key
api_key_path = Path("../../openai_po991_arc.key")
os.environ["OPENAI_API_KEY"] = api_key_path.read_text().strip()
client = OpenAI()

def get_llm_response(conversation):
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=conversation,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    save_dir = Path('./result')

    with open("data/larc_gpt4.json") as json_file:
        larc_gpt4 = json.load(json_file)
    selected_tasks = larc_gpt4[:1]

    for task in selected_tasks:
        conv = Coversation(save_dir=save_dir)
        few_shot_id = 0
        target_id = few_shot_id + 1
        prompt = generate_nl_and_io_prompt(task, few_shot_id, target_id)
        conv.add_user(prompt)
        conv.print()
        conv.add_assistant(get_llm_response(conv.history))
        conv.print()

        scores = []
        for round in range(2):
            print(f"Round {round}")
            conv.add_user(generate_review_prompt())
            conv.print()
            answer = get_llm_response(conv.history)
            conv.add_assistant(answer)
            conv.print()

            pred_grid = extract_output(answer, "output_grid")
            gt_grid = task["problem"]["train"][target_id]["output"]

            score = eval_score(pred_grid, gt_grid)
            scores.append(score)
            if score == 1:
                print("Get the correct answer!")
                print("Break the round loop")
                break
    
        # テストに対する推論
        conv.add_user(generate_test_prediction_prompt(task))
        conv.print()
        answer = get_llm_response(conv.history)
        conv.add_assistant(answer)
        pred_grid = extract_output(answer, "output_grid")
        gt_grid = task["problem"]["test"][0]["output"]
        score = eval_score(pred_grid, gt_grid)
        print(f"Test score: {score}")
        conv.print()

        conv.save_conversation()