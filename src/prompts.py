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


def generate_nl_and_io_prompt(task, few_shot_id, target_id):

    instruction = "here is the instruction of how to transform the grid: \n"
    instruction += (
        task["description"]["description_input"]
        + task["description"]["description_output_grid_size"]
        + task["description"]["description_output"]
    )

    train_input = task["problem"]["train"][few_shot_id]["input"]
    train_output = task["problem"]["train"][few_shot_id]["output"]
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

    input_grid = task["problem"]["train"][target_id]["input"]

    prompt = (
        preamble
        + instruction
        + input_output_example
        + "\n\nThe input grid is:\n"
        + str(input_grid)
        + "\n\nWhat is the output grid?"
        + "\n\nOutput grid should be surrounded by <output_grid> and <output_grid>"
    )
    return prompt


# タスクを入力したら、テストに対して推論をする関数
def generate_test_prediction_prompt(task):

    input_grid = task["problem"]["test"][0]["input"]

    # NOTE: preamble いらないかも
    prompt = (
        preamble
        + "\n\nThe input grid is:\n"
        + str(input_grid)
        + "\n\nWhat is the output grid?"
        + "\n\nOutput grid should be surrounded by <output_grid> and <output_grid>"
    )
    return prompt


def calc_diff(pred_grid, gt_grid):
    diff = gt_grid != pred_grid
    diff = diff.astype(int)
    return diff


def generate_review_prompt(review_type, pred_grid, gt_grid):

    if review_type == "type1":
        prompt = f"""Your output was incorrect.
Please clearly identify the differences between the correct answer and your output.
Specifically, highlight which part of the given task description was not accurately executed, resulting in this discrepancy.
Then, provide the correct answer based on the correct interpretation of the task.
Output grid should be surrounded by <output_grid> and <output_grid>.
"""

    elif review_type == "type2":
        if pred_grid is None:
            prompt = f"""Your output was incorrect.
First of all, the grid shape is not correct.
Please clearly identify the differences between the correct answer and your output.
Specifically, highlight which part of the given task description was not accurately executed, resulting in this discrepancy.
Then, provide the correct answer based on the correct interpretation of the task.
Output grid should be surrounded by <output_grid> and <output_grid>.
    """
        else:
            diff = str(calc_diff(pred_grid, gt_grid))
            prompt = f"""Your output was incorrect.
The difference is 
{diff},
where 1 refers to the incorrect areas and 0 correct.
    
The incorrect areas have been identified and should be analysed again and corrected appropriately based on this.
Please clearly identify the differences between the correct answer and your output.
Then, provide the correct answer based on the correct interpretation of the task.
Output grid should be surrounded by <output_grid> and <output_grid>.
    """

    return prompt
