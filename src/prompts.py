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
def nl_and_io_prompt(task, few_shot_id, target_id):
    
    instruction = "here is the instruction of how to transform the grid: \n"
    instruction += task['description']['description_input'] + task['description']['description_output_grid_size'] + task['description']['description_output']
    
    train_input = task['problem']['train'][few_shot_id]['input']
    train_output = task['problem']['train'][few_shot_id]['output']
    input_output_example = "\n\nhere is an example of an input grid and its corresponding output grid:\n"
    input_output_example += "example input grid:\n" + str(train_input) + "\nexample output grid:\n" + str(train_output) + "\n\n"

    input_grid = task['problem']['train'][target_id]['input']

    prompt = preamble + instruction + input_output_example + "\n\nThe input grid is:\n" + str(input_grid) + "\n\nWhat is the output grid?" + "\n\nOutput gird surrounded by <output_grid> and <output_grid>"
    return prompt 

def review(grid):
    prompt = f'''Your output was incorrect.

Please clearly identify the differences between the correct answer and your output.
Specifically, highlight which part of the given task description was not accurately executed, resulting in this discrepancy.
Then, provide the correct answer based on the correct interpretation of the task.
Output gird surrounded by <output_grid> and <output_grid>
'''
    return prompt