# the experiment ! get gpt4 in here
from openai import OpenAI
import json
import numpy as np
from src.prompts import nl_and_io_prompt, review

# OpenAI の ver 1.0 以降でLLMの出力の取得
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

    
if __name__ == '__main__':

    max_num_tasks = 1

    # open data/larc_gpt4.json
    with open('data/larc_gpt4.json') as json_file:
        larc_gpt4 = json.load(json_file)

    print(len(larc_gpt4))
    for task in larc_gpt4[:max_num_tasks]:

        conversation = [{"role": "system", "content": "You are an autonomous task solver."}]

        # TODO: few_shot_id と target_id は len(task['problem']['train']) に対する for 文を回して順次指定する 
        few_shot_id = 0
        target_id = few_shot_id + 1
        prompt = nl_and_io_prompt(task, few_shot_id, target_id)
        conversation.append({'role': 'user', 'content': prompt})

        # print(prompt)
        answer = get_llm_response(conversation)
        # print(answer)
        conversation.append({'role': 'assistant', 'content': answer})

        for round in range(1):
            print(f'Round {round}')
            prompt = review(task['problem']['train'][1]['output'])
            conversation.append({'role': 'user', 'content': prompt})

            answer = get_llm_response(conversation)
            conversation.append({'role': 'assistant', 'content': answer})

            pred_grid = answer.split('<output_grid>')[1].split('</output_grid>')[0]
            gt_grid = task['problem']['train'][1]['output']

            score = eval_score(pred_grid, gt_grid)
            print(f'score: {score}')

            if score == 1:
                break

        task['prediction'] = pred_grid

        for conv in conversation:
            print('-'*10)
            print(f"[{conv['role']}]")
            print(conv['content'])

        with open('data/larc_gpt4_newer.json', 'w') as outfile:
            json.dump(larc_gpt4, outfile)