import pandas as pd
import random
import os
import json

QA_FILE = "python_choice_questions.csv"
WRONG_FILE = "wrong_choice.json"

def load_qa():
    df = pd.read_csv(QA_FILE)
    questions = df.to_dict(orient="records")
    return questions

def load_wrong():
    if os.path.exists(WRONG_FILE):
        with open(WRONG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_wrong(wrong_dict):
    with open(WRONG_FILE, 'w', encoding='utf-8') as f:
        json.dump(wrong_dict, f, ensure_ascii=False, indent=2)

def choose_question(questions, wrong_dict):
    pool = []
    for i, q in enumerate(questions):
        weight = 1 + wrong_dict.get(str(i), 0) * 3
        pool += [i] * weight
    idx = random.choice(pool)
    return idx, questions[idx]

def main():
    questions = load_qa()
    wrong_dict = load_wrong()
    options = ['A', 'B', 'C', 'D']

    while True:
        idx, q = choose_question(questions, wrong_dict)
        print("\n题目：", q['题目'])

        # 随机打乱选项顺序
        items = [(opt, q[opt]) for opt in options if pd.notnull(q[opt])]
        random.shuffle(items)

        # 生成新的选项字母映射
        new_opt_map = {new_opt: val for new_opt, (old_opt, val) in zip(options, items)}
        # 找到正确答案新字母
        right_val = q[q['答案']]
        for k, v in new_opt_map.items():
            if v == right_val:
                right_ans = k

        for opt in options:
            if opt in new_opt_map:
                print(f"{opt}. {new_opt_map[opt]}")

        user = input("你的答案（A/B/C/D，q退出）：").strip().upper()
        print(f"标准答案：{right_ans}")
        if user == 'Q':
            break
        if user == right_ans:
            print("✅ 正确！")
            if str(idx) in wrong_dict:
                wrong_dict[str(idx)] = max(wrong_dict[str(idx)] - 1, 0)
        else:
            print("❌ 错误，已提升此题权重，下次更容易抽到")
            wrong_dict[str(idx)] = wrong_dict.get(str(idx), 0) + 1
        save_wrong(wrong_dict)

if __name__ == '__main__':
    main()