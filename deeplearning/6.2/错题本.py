import pandas as pd
import random
import json
import os
import datetime
import io
import contextlib
from flask import Flask, render_template_string, request, session

app = Flask(__name__)
app.secret_key = 'super_secret'

QA_FILE = "python_choice_questions.csv"
WRONG_FILE = "wrong_choice.json"
PROGRESS_FILE = "progress.json"
MASTER_THRESHOLD = 3


def load_qa():
    df = pd.read_csv(QA_FILE)
    return df


def load_wrong():
    if os.path.exists(WRONG_FILE):
        with open(WRONG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_wrong(wrong_dict):
    with open(WRONG_FILE, 'w', encoding='utf-8') as f:
        json.dump(wrong_dict, f, ensure_ascii=False, indent=2)


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def get_today_str():
    return datetime.datetime.now().strftime('%Y-%m-%d')


def get_available_indices(df, progress):
    today = get_today_str()
    available = []
    for i in range(len(df)):
        info = progress.get(str(i), {"streak": 0, "last_date": ""})
        if not (info["streak"] >= MASTER_THRESHOLD and info["last_date"] == today):
            available.append(i)
    return available


# 安全执行用户代码并捕获输出（适合基础 print 类题，勿做危险命令）
def safe_run(user_code):
    allowed_builtins = {'print': print, 'range': range, 'len': len, 'str': str, 'int': int, 'float': float,
                        'list': list}
    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f):
            exec(user_code, {'__builtins__': allowed_builtins})
    except Exception as e:
        return f"【运行错误】{e}"
    return f.getvalue().strip()


@app.route("/", methods=["GET", "POST"])
def index():
    df = load_qa()
    wrong_dict = load_wrong()
    progress = load_progress()
    today = get_today_str()

    available = get_available_indices(df, progress)
    if not available:
        return render_template_string("""
            <div style="margin:60px auto;max-width:430px;background:#fff;border-radius:18px;box-shadow:0 4px 22px #d7ebfa;padding:40px 36px;">
            <h3 style="color:#28b58b;">🎉 今日任务已完成！</h3>
            <p style="margin-top:24px;">所有题目已达到熟练度阈值({{threshold}}次)，欢迎明天继续巩固～</p>
            <a href="/" class="btn btn-primary mt-3">刷新</a>
            </div>
            <div style="text-align:center;margin-top:42px;color:#aaa;font-size:14px;">by Python刷题小助手</div>
        """, threshold=MASTER_THRESHOLD)

    if 'idx' not in session or request.method == "GET":
        idx = random.choice(available)
        session['idx'] = idx
        feedback = None
        correct = None
        user_input = ""
    else:
        idx = session['idx']
        feedback = None
        correct = None
        user_input = ""
        user_ans = request.form.get("answer", "").strip()
        user_input = user_ans
        real_ans = str(df.iloc[idx]['答案']).strip()
        typ = str(df.iloc[idx]['类型']).strip()
        std_output = str(df.iloc[idx]['标准输出']) if '标准输出' in df.columns else ""
        # 选择题
        if typ == "choice":
            is_right = user_ans.upper() == real_ans.upper()
            if is_right:
                feedback = "✅ 恭喜你，答对了！"
                correct = True
                info = progress.get(str(idx), {"streak": 0, "last_date": ""})
                if info["last_date"] == today:
                    info["streak"] += 1
                else:
                    info["streak"] = 1
                info["last_date"] = today
                progress[str(idx)] = info
                save_progress(progress)
                if str(idx) in wrong_dict:
                    wrong_dict[str(idx)] = max(wrong_dict[str(idx)] - 1, 0)
            else:
                feedback = "❌ 答错了，已重置熟练度，提升错题权重！"
                correct = False
                progress[str(idx)] = {"streak": 0, "last_date": today}
                save_progress(progress)
                wrong_dict[str(idx)] = wrong_dict.get(str(idx), 0) + 1
            save_wrong(wrong_dict)
        # 概念题
        elif typ == "concept":
            feedback = f"""<div class="markdown-body">【你的答案】{user_ans}</div><br><b>参考答案：</b><div class="markdown-body">{df.iloc[idx]['参考答案']}</div>"""
            correct = None
            info = progress.get(str(idx), {"streak": 0, "last_date": ""})
            info["streak"] += 1
            info["last_date"] = today
            progress[str(idx)] = info
            save_progress(progress)
        # 代码题（自动判分：输出判定）
        elif typ == "code":
            std_output = std_output.replace('\\n', '\n').strip()
            user_out = safe_run(user_ans)
            if std_output:
                if user_out == std_output:
                    feedback = f"""✅输出正确！\n<div class="markdown-body"><pre style='background:#f6fff6;padding:6px;'>{user_out}</pre></div>"""
                    correct = True
                else:
                    feedback = (
                        f"""❌输出不一致！<br><b>你的输出：</b>\n<div class="markdown-body"><pre style='background:#fff7f6;padding:6px;'>{user_out}</pre></div>"""
                        f"""<b>标准输出：</b>\n<div class="markdown-body"><pre style='background:#f8fafb;padding:6px;'>{std_output}</pre></div>"""
                    )
                    correct = False
            else:
                feedback = (
                    f"""<div class="markdown-body">【你的代码】<pre style='background:#f5f7fa;border-radius:6px;padding:10px'>{user_ans}</pre></div>"""
                    f"""<b>参考代码：</b><div class="markdown-body"><pre style='background:#f8fafb;border-radius:6px;padding:10px'>{df.iloc[idx]['代码内容']}</pre></div>"""
                )
                correct = None
            info = progress.get(str(idx), {"streak": 0, "last_date": ""})
            if correct:
                info["streak"] += 1
            else:
                info["streak"] = 0
            info["last_date"] = today
            progress[str(idx)] = info
            save_progress(progress)

    q = df.iloc[session['idx']]
    opts = ['A', 'B', 'C', 'D']
    mastered_count = sum(1 for v in progress.values() if v["streak"] >= MASTER_THRESHOLD and v["last_date"] == today)
    total = len(df)
    return render_template_string(HTML, q=q, opts=opts, feedback=feedback, correct=correct, user_input=user_input,
                                  mastered_count=mastered_count, total=total, threshold=MASTER_THRESHOLD)


@app.route("/add", methods=["GET", "POST"])
def add():
    message = ""
    if request.method == "POST":
        typ = request.form.get("type")
        ques = request.form.get("ques", "").strip()
        A = request.form.get("A", "").strip()
        B = request.form.get("B", "").strip()
        C = request.form.get("C", "").strip()
        D = request.form.get("D", "").strip()
        ans = request.form.get("ans", "").strip().upper()
        ref_ans = request.form.get("ref_ans", "").strip()
        code_content = request.form.get("code_content", "").strip()
        std_output = request.form.get("std_output", "").strip()

        # 必填校验
        if typ == "choice":
            ok = ques and A and B and C and D and ans in ['A', 'B', 'C', 'D']
        elif typ == "concept":
            ok = ques and ref_ans
        elif typ == "code":
            ok = ques and code_content
        else:
            ok = False
        if not ok:
            message = "请填写所有必填项"
        else:
            df = pd.read_csv(QA_FILE)
            row = {
                "题目": ques, "A": A, "B": B, "C": C, "D": D, "答案": ans,
                "类型": typ, "参考答案": ref_ans, "代码内容": code_content, "标准输出": std_output
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(QA_FILE, index=False, encoding='utf-8')
            message = "添加成功！"
    return render_template_string(ADD_HTML, message=message)


HTML = """
<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>多题型刷题系统</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/simplemde/latest/simplemde.min.css">
  <script src="https://cdn.jsdelivr.net/simplemde/latest/simplemde.min.js"></script>
  <style>
    body { background: #f6fbff; min-height:100vh;}
    .quiz-container { max-width: 530px; margin:48px auto 0;background: #fff; border-radius:22px; box-shadow: 0 8px 32px rgba(0,60,180,0.13), 0 1.5px 3px #a1b2c4; padding:36px 32px 28px 32px;}
    .quiz-title { font-size:2rem;font-weight:700;text-align:center;letter-spacing:2px;color:#2759e0;margin-bottom:18px;}
    .progress-info { font-size:1.08rem;color:#199e5b;margin-bottom:10px;text-align:center;}
    .question { font-size:1.13rem;font-weight:600;color:#273149;margin-bottom:2.1rem;letter-spacing:0.5px;}
    .option-btn { width:100%;margin-bottom:17px;font-size:1.04rem;text-align:left;border-radius:13px;box-shadow:0 1px 2.5px #e9f1ff;transition:0.15s;padding:10px 16px;border:2px solid #e7efff;background:#f7fafd;}
    .option-btn.selected,.option-btn:focus {border:2px solid #2c8be8 !important; background:#e6f4ff;}
    .option-btn[disabled] {color:#95a1b3 !important;background:#f3f3f3;border-color:#dde2e8 !important;}
    .form-control,.form-select{border-radius:10px;}
  </style>
</head>
<body>
  <div class="quiz-container">
    <div class="quiz-title">多题型刷题系统</div>
    <div class="progress-info">今日已掌握 {{ mastered_count }}/{{ total }} 题（连续答对 {{ threshold }} 次为掌握）</div>
    <div class="question">{{ q['题目'] }}</div>
    <form method="POST" id="quiz-form">
      {% if q['类型']=="choice" %}
        {% for opt in opts %}
        <button type="submit" name="answer" value="{{ opt }}" class="btn option-btn {% if request.form.get('answer') == opt %}selected{% endif %}" {% if feedback %}disabled{% endif %}>
          <b>{{ opt }}.</b> {{ q[opt] }}
        </button>
        {% endfor %}
      {% elif q['类型']=="concept" %}
        <textarea id="markdown-editor" class="form-control mb-3" name="answer" rows=2 placeholder="请输入你的答案" {% if feedback %}disabled{% endif %}>{{ user_input }}</textarea>
        <button type="submit" class="btn btn-primary w-100" {% if feedback %}disabled{% endif %}>提交</button>
      {% elif q['类型']=="code" %}
        <textarea id="markdown-editor" class="form-control mb-3" name="answer" rows=6 placeholder="请输入完整代码，支持多行" {% if feedback %}disabled{% endif %}>{{ user_input }}</textarea>
        <button type="submit" class="btn btn-primary w-100" {% if feedback %}disabled{% endif %}>提交</button>
      {% endif %}
      {% if feedback %}
        <div class="alert alert-info mt-3" style="white-space:pre-wrap;">{{ feedback|safe }}</div>
        <a href="{{ url_for('index') }}" class="btn btn-success w-100 mt-2">下一题</a>
      {% endif %}
    </form>
        <div class="text-center mt-4">
        <a href="/add" class="btn btn-outline-secondary">➕ 添加新题</a>
        </div>
  </div>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/github-markdown-css/github-markdown.min.css">
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script>
  if (document.getElementById('markdown-editor')) {
    var simplemde = new SimpleMDE({ 
      element: document.getElementById('markdown-editor'),
      status: false,
      spellChecker: false,
      autofocus: true,
      placeholder: "请在此输入支持Markdown的答案或代码"
    });
    // 表单提交时把Markdown内容同步到textarea
    document.getElementById('quiz-form').onsubmit = function() {
      document.getElementById('markdown-editor').value = simplemde.value();
    }
  }
  document.querySelectorAll('.markdown-body').forEach(el => {
    el.innerHTML = marked.parse(el.textContent);
  });
  </script>
</body>
</html>
"""

ADD_HTML = """
<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>添加新题</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/simplemde/latest/simplemde.min.css">
  <script src="https://cdn.jsdelivr.net/simplemde/latest/simplemde.min.js"></script>
  <style>
    body { background: #f6fbff; min-height:100vh;}
    .quiz-container { max-width: 480px; margin:48px auto 0;background: #fff; border-radius:22px; box-shadow: 0 8px 32px rgba(0,60,180,0.13), 0 1.5px 3px #a1b2c4; padding:36px 32px 28px 32px;}
    .quiz-title { font-size:2rem;font-weight:700;text-align:center;letter-spacing:2px;color:#2759e0;margin-bottom:18px;}
    .form-label{font-weight:600;}
  </style>
  <script>
  function onTypeChange() {
    var t = document.getElementById('type').value;
    document.getElementById('choice').style.display = (t=='choice') ? 'block':'none';
    document.getElementById('concept').style.display = (t=='concept') ? 'block':'none';
    document.getElementById('code').style.display = (t=='code') ? 'block':'none';
  }
  </script>
</head>
<body onload="onTypeChange()">
  <div class="quiz-container">
    <div class="quiz-title">添加新题</div>
    <form method="POST">
      <div class="mb-3">
        <label class="form-label">题型</label>
        <select name="type" id="type" class="form-select" onchange="onTypeChange()">
          <option value="choice">选择题</option>
          <option value="concept">简答题</option>
          <option value="code">代码题</option>
        </select>
      </div>
      <div class="mb-3">
        <label class="form-label">题目</label>
        <textarea name="ques" class="form-control" placeholder="题目内容" rows=3></textarea>
      </div>
      <div id="choice" style="display:none;">
        <div class="mb-3"><input name="A" class="form-control" placeholder="选项A"></div>
        <div class="mb-3"><input name="B" class="form-control" placeholder="选项B"></div>
        <div class="mb-3"><input name="C" class="form-control" placeholder="选项C"></div>
        <div class="mb-3"><input name="D" class="form-control" placeholder="选项D"></div>
        <div class="mb-3">
          <select name="ans" class="form-select">
            <option value="">请选择正确答案</option>
            <option value="A">A</option><option value="B">B</option>
            <option value="C">C</option><option value="D">D</option>
          </select>
        </div>
      </div>
      <div id="concept" style="display:none;">
        <div class="mb-3">
          <label class="form-label">参考答案（支持Markdown）</label>
          <textarea id="concept-md" name="ref_ans" class="form-control" rows=4 placeholder="请输入参考答案"></textarea>
        </div>
      </div>
      <div id="code" style="display:none;">
        <div class="mb-3">
          <label class="form-label">代码内容（支持Markdown）</label>
          <textarea id="code-md" name="code_content" class="form-control" rows=6 placeholder="请输入标准代码"></textarea>
        </div>
        <div class="mb-3">
          <label class="form-label">标准输出</label>
          <input name="std_output" class="form-control" placeholder="请填写该代码的预期标准输出（可选）">
        </div>
      </div>
      {% if message %}
      <div class="alert alert-info">{{ message }}</div>
      {% endif %}
      <button type="submit" class="btn btn-primary w-100">保存题目</button>
      <a href="/" class="btn btn-secondary w-100 mt-2">返回首页</a>
    </form>
  </div>
  <script>
  if (document.getElementById('concept-md')) {
    var mde_concept = new SimpleMDE({ 
      element: document.getElementById('concept-md'),
      status: false,
      spellChecker: false,
      autofocus: false,
      placeholder: "请在此输入支持Markdown的参考答案"
    });
    document.querySelector("form").onsubmit = function() {
      document.getElementById('concept-md').value = mde_concept.value();
    }
  }
  if (document.getElementById('code-md')) {
    var mde_code = new SimpleMDE({ 
      element: document.getElementById('code-md'),
      status: false,
      spellChecker: false,
      autofocus: false,
      placeholder: "请在此输入支持Markdown的代码内容"
    });
    document.querySelector("form").onsubmit = function() {
      document.getElementById('code-md').value = mde_code.value();
    }
  }
  </script>
</body>
</html>
"""
if __name__ == "__main__":
    app.run(debug=True)
