import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from search_and_load import auto_build_docs

transformer = SentenceTransformer('shibing624/text2vec-base-chinese')


def build_prompt(query, result):
    context = '\n'.join(result)
    prompt = f"""
你是一个专业的中文AI助手，能够准确理解中文表达中的情感细微差别。

请分析以下用户问题，结合提供的资料进行回答。

资料：
{context}

用户问题：
{query}

**重要判断规则：**
1. 注意区分“感动得哭了”“幸福得流泪”这样的正面情感与“伤心哭了”这种负面情感。
2. 出现“哭”不代表一定是“悲伤”，要结合上下文判断。
3. 如果表达的是感动、感激、被打动，这是正面情绪，更偏向“喜悦”或“正面”。
4. 仅当表达的是失落、失望、痛苦，才归为“悲伤”或“负面”。
5. 没有明显情绪色彩的，归类为“中性”。
6. 请先详细思考，再输出分类或解答。

如果资料中没有答案，请直接回复“资料中未找到相关信息”。
"""
    return prompt


def ask_ollama(prompt):
    response = ollama.chat(
        model="deepseek-r1:1.5b",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    return response['message']['content']


def rag_chat(query):
    docs = auto_build_docs(query)

    if not docs:
        return " 未找到相关网页内容，无法回答。"

    embeddings = transformer.encode(docs)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))

    query_embedding = transformer.encode(query)
    D, I = index.search(query_embedding.reshape(1, -1), 3)
    result = [docs[i] for i in I[0] if i < len(docs)]

    prompt = build_prompt(query, result)
    answer = ask_ollama(prompt)

    return f'[{query}]的答案是：\n{answer}'