import csv
import argparse
import openai
from tqdm import tqdm
import re
import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'utils', 'config4all.json')


def load_dataset(path: str):
    data = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            question = row[0].strip()
            choices = row[1:5]
            answer = row[5].strip()
            data.append((question, choices, answer))
    return data


def build_prompt(question: str, choices: list[str]) -> str:
    letters = ['A', 'B', 'C', 'D']
    lines = [question]
    for l, c in zip(letters, choices):
        lines.append(f"{l}. {c}")
    lines.append('\nChoose the correct option. Respond with the final answer in the form \\boxed{X}, where X is A, B, C or D.')
    return '\n'.join(lines)


def query(messages: list[dict], model: str, api_key: str) -> str:
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        api_key=api_key,
    )
    return resp['choices'][0]['message']['content']


def run_debate(question: str, choices: list[str], model: str, api_key: str) -> str:
    topic = build_prompt(question, choices)
    config = json.load(open(CONFIG_PATH))
    config['debate_topic'] = topic

    def round_word(num: int) -> str:
        mapping = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth',
            6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return mapping[num]

    aff_msgs = [{'role': 'system', 'content': config['player_meta_prompt']}]
    neg_msgs = [{'role': 'system', 'content': config['player_meta_prompt']}]
    mod_msgs = [{'role': 'system', 'content': config['moderator_meta_prompt']}]

    # First round
    aff_msgs.append({'role': 'user', 'content': config['affirmative_prompt']})
    aff_ans = query(aff_msgs, model, api_key)
    aff_msgs.append({'role': 'assistant', 'content': aff_ans})

    neg_msgs.append({'role': 'user', 'content': config['negative_prompt'].replace('##aff_ans##', aff_ans)})
    neg_ans = query(neg_msgs, model, api_key)
    neg_msgs.append({'role': 'assistant', 'content': neg_ans})

    mod_msgs.append({'role': 'user', 'content': config['moderator_prompt'].replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans).replace('##round##', 'first')})
    mod_ans = query(mod_msgs, model, api_key)
    mod_msgs.append({'role': 'assistant', 'content': mod_ans})
    result = eval(mod_ans)

    round_num = 2
    while not result.get('debate_answer') and round_num <= 3:
        aff_msgs.append({'role': 'user', 'content': config['debate_prompt'].replace('##oppo_ans##', neg_ans)})
        aff_ans = query(aff_msgs, model, api_key)
        aff_msgs.append({'role': 'assistant', 'content': aff_ans})

        neg_msgs.append({'role': 'user', 'content': config['debate_prompt'].replace('##oppo_ans##', aff_ans)})
        neg_ans = query(neg_msgs, model, api_key)
        neg_msgs.append({'role': 'assistant', 'content': neg_ans})

        mod_msgs.append({'role': 'user', 'content': config['moderator_prompt'].replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans).replace('##round##', round_word(round_num))})
        mod_ans = query(mod_msgs, model, api_key)
        mod_msgs.append({'role': 'assistant', 'content': mod_ans})
        result = eval(mod_ans)
        round_num += 1

    if not result.get('debate_answer'):
        judge_msgs = [
            {'role': 'system', 'content': config['moderator_meta_prompt']},
            {'role': 'user', 'content': config['judge_prompt_last1'].replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans)}
        ]
        ans = query(judge_msgs, model, api_key)
        judge_msgs.append({'role': 'assistant', 'content': ans})
        judge_msgs.append({'role': 'user', 'content': config['judge_prompt_last2']})
        ans = query(judge_msgs, model, api_key)
        judge_msgs.append({'role': 'assistant', 'content': ans})
        result = eval(ans)

    return result.get('debate_answer', '')


def extract_choice(text: str) -> str:
    m = re.search(r"\\boxed{\s*([A-D])\s*}", text)
    if m:
        return m.group(1)
    m = re.search(r"([A-D])", text)
    return m.group(1) if m else ''


def evaluate(data, model: str, api_key: str):
    correct = 0
    for q, choices, ans in tqdm(data):
        reply = run_debate(q, choices, model, api_key)
        pred = extract_choice(reply)
        if pred == ans:
            correct += 1
    return correct / len(data)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on college mathematics test")
    parser.add_argument('-k', '--api-key', required=True, help='OpenAI API key')
    parser.add_argument('-m', '--model', default='gpt-3.5-turbo', help='Model name')
    parser.add_argument('-d', '--dataset', default='data/CollegeMath/college_mathematics_test.csv', help='Dataset path')
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    acc = evaluate(data, args.model, args.api_key)
    print(f"Accuracy: {acc:.2%}")


if __name__ == '__main__':
    main()
