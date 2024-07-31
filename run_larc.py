# the experiment ! get gpt4 in here
import os
from openai import OpenAI
import json
import numpy as np

# OpenAI の ver 1.0 以降でLLMの出力の取得
api_key_path = "../../openai_po991_arc.key"
with open(api_key_path) as f:
    api_key = f.readline()[:-1]
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI()


def get_llm_response(conversation):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation,
    )
    return response.choices[0].message.content


def eval_score(pred_grid, gt_grid):
    print(pred_grid)
    print(gt_grid)

    pred_grid = np.array(eval(str(pred_grid)))
    gt_grid = np.array(eval(str(gt_grid)))

    pred_shape = pred_grid.shape
    gt_shape = gt_grid.shape

    print(pred_shape)
    print(gt_shape)

    if pred_shape != gt_shape:
        return 0
    else:
        score = (pred_grid == gt_grid).sum() / (pred_shape[0] * pred_shape[1])
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


if __name__ == "__main__":

    max_num_tasks = 1

    # open data/larc_gpt4.json
    with open("data/larc_gpt4.json") as json_file:
        larc_gpt4 = json.load(json_file)

    print(len(larc_gpt4))
    for task in larc_gpt4[:max_num_tasks]:

        conversation = [
            {"role": "system", "content": "You are an autonomous task solver."}
        ]
        prompt = nl_and_io_prompt(task)
        conversation.append({"role": "user", "content": prompt})

        # print(prompt)
        answer = get_llm_response(conversation)
        # print(answer)
        conversation.append({"role": "assistant", "content": answer})

        for round in range(1):
            print(f"Round {round}")
            prompt = review(task["problem"]["train"][1]["output"])
            conversation.append({"role": "user", "content": prompt})

            answer = get_llm_response(conversation)
            conversation.append({"role": "assistant", "content": answer})

            pred_grid = answer.split("<output_grid>")[1].split("</output_grid>")[0]
            gt_grid = task["problem"]["train"][1]["output"]

            score = eval_score(pred_grid, gt_grid)
            print(f"score: {score}")

            if score == 1:
                break

        task["prediction"] = pred_grid

        for conv in conversation:
            print("-" * 10)
            print(f"[{conv['role']}]")
            print(conv["content"])

        with open("data/larc_gpt4_newer.json", "w") as outfile:
            json.dump(larc_gpt4, outfile)
