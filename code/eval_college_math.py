import csv
import argparse
import openai
from tqdm import tqdm
import re


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


def ask(model: str, api_key: str, prompt: str):
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        api_key=api_key,
    )
    return resp['choices'][0]['message']['content']


def extract_choice(text: str) -> str:
    m = re.search(r"\\boxed{\s*([A-D])\s*}", text)
    if m:
        return m.group(1)
    m = re.search(r"([A-D])", text)
    return m.group(1) if m else ''


def evaluate(data, model: str, api_key: str):
    correct = 0
    for q, choices, ans in tqdm(data):
        prompt = build_prompt(q, choices)
        reply = ask(model, api_key, prompt)
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
