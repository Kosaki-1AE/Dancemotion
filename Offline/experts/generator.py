# generator.py — 日本語“仕様メモ”→コード自動生成（CLI/WEBAPI）
# 使い方: python3 generator.py spec.md out_dir

import sys, re, os, json, textwrap
from pathlib import Path

SECTIONS = ["タスク", "環境", "入力", "出力", "手順", "エンドポイント", "例", "依存"]

# --- ユーティリティ ---
def read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

# --- Spec 解析（# 見出し or キー: 値 or 箇条書き 対応） ---
def parse_spec(spec_text: str) -> dict:
    spec = {k: [] for k in SECTIONS}

    # 1) # 見出し での分割
    current = None
    for line in spec_text.splitlines():
        t = line.strip()
        m = re.match(r"^#\s*(.+)$", line)
        if m:
            title = m.group(1).strip()
            for s in SECTIONS:
                if s in title:
                    current = s
                    break
            continue
        # キー: 値 形式
        m2 = re.match(r"^(\S+?)\s*:\s*(.+)$", t)
        if m2 and m2.group(1) in SECTIONS:
            current = m2.group(1)
            spec[current].append(m2.group(2))
            continue
        if current in SECTIONS and t:
            spec[current].append(t)

    # 2) 文字列化
    spec2 = {}
    for k, vs in spec.items():
        vs2 = [v for v in vs if v and not v.startswith("//")]
        spec2[k] = "\n".join(vs2).strip()
    return spec2

# --- 環境の判定 ---
def detect_env(spec: dict) -> str:
    env = (spec.get("環境") or "").lower()
    if "web" in env or "api" in env:
        return "webapi"
    return "cli"

# --- 手順の自然語 → 操作ブロック（簡易ルール） ---
# 代表的な“やりたいこと”をパターン認識してコード断片へ。
# 未知は TODO コメント化。

def op_from_line(line: str) -> dict:
    L = line.strip("・-• ")
    # 小文字化
    if re.search(r"小文字", L):
        return {"op": "lower"}
    # 大文字化
    if re.search(r"大文字", L):
        return {"op": "upper"}
    # 行数を数える
    if re.search(r"行数|行を数", L):
        return {"op": "count_lines"}
    # 重複削除（行ベース）
    if re.search(r"重複.*削", L):
        return {"op": "unique_lines"}
    # 正規表現抽出: 例) 正規表現 /\d+/ を抽出
    m = re.search(r"正規表現\s*(/.*?/)", L)
    if m:
        return {"op": "regex_extract", "pattern": m.group(1)[1:-1]}
    # JSON キー抽出: 例) JSONのキー name を1行ずつ出力
    m = re.search(r"JSON.*キー\s*(\w+)", L)
    if m:
        return {"op": "json_key", "key": m.group(1)}
    # CSVの列抽出: 例) CSVの列 email を出力
    m = re.search(r"CSV.*列\s*(\w+)", L)
    if m:
        return {"op": "csv_col", "col": m.group(1)}
    # 数値合計
    if re.search(r"数値.*合計|合計する", L):
        return {"op": "sum_numbers"}
    return {"op": "todo", "line": L}


def parse_steps(spec: dict) -> list:
    steps = []
    text = spec.get("手順", "")
    for line in text.splitlines():
        if line.strip():
            steps.append(op_from_line(line))
    return steps

# --- CLI 生成 ---
CLI_TMPL = """
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, re, json, csv


def read_stdin() -> str:
    return sys.stdin.read()


def main():
    data = read_stdin()
    out_lines = []
    # === 自動生成ステップ ===
{steps}
    # 出力
    sys.stdout.write("\n".join(out_lines) if out_lines else data)

if __name__ == "__main__":
    main()
"""

# 各ステップのコード断片（インデント4）
STEP_SNIPPETS = {
    "lower": """
    data = data.lower()
    """,
    "upper": """
    data = data.upper()
    """,
    "count_lines": """
    out_lines = [str(len(data.splitlines()))]
    """,
    "unique_lines": """
    seen = set(); unique = []
    for ln in data.splitlines():
        if ln not in seen:
            seen.add(ln); unique.append(ln)
    data = "\n".join(unique)
    """,
    "regex_extract": """
    pat = re.compile({pattern!r})
    out_lines = pat.findall(data)
    """,
    "json_key": """
    out_lines = []
    for ln in data.splitlines():
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
            if {key!r} in obj:
                out_lines.append(str(obj[{key!r}]))
        except Exception:
            pass
    """,
    "csv_col": """
    out_lines = []
    import io
    rdr = csv.DictReader(io.StringIO(data))
    for row in rdr:
        if {col!r} in row:
            out_lines.append(row[{col!r}])
    """,
    "sum_numbers": """
    import math
    nums = re.findall(r"-?\\d+(?:\\.\\d+)?", data)
    s = sum(float(x) for x in nums)
    out_lines = [str(s)]
    """,
    "todo": """
    # TODO: 手で実装: {line}
    # ここに追記してね
    """,
}


def render_cli(steps: list) -> str:
    parts = []
    for st in steps:
        op = st["op"]
        code = STEP_SNIPPETS[op]
        code = code.format(**{k: v for k, v in st.items() if k != "op"})
        parts.append(textwrap.indent(code.strip("\n"), "    "))
    return CLI_TMPL.format(steps="\n".join(parts))

# --- Web API 生成（FastAPI） ---
WEB_TMPL = """
# main.py — FastAPI 自動生成
from fastapi import FastAPI, Query, Body
from pydantic import BaseModel
import re, json

app = FastAPI(title="Spec→Code AutoAPI")

{endpoints}
"""

# エンドポイント定義の例：
# - GET /sum?x&y -> x+y
# - POST /echo (json) -> 同じjson
EP_GET_SUM = """
@app.get("/sum")
def sum_xy(x: float = Query(...), y: float = Query(...)):
    return {"result": x + y}
"""

EP_POST_ECHO = """
class EchoModel(BaseModel):
    payload: dict

@app.post("/echo")
def echo(m: EchoModel):
    return {"echo": m.payload}
"""

# ルールベース（最小）

def ep_from_line(line: str) -> str:
    L = line.strip("•-・ ")
    # GET /sum?x&y -> x+y
    if re.search(r"GET\s+/sum", L):
        return EP_GET_SUM
    # POST /echo (json) -> 同じjson
    if re.search(r"POST\s+/echo", L):
        return EP_POST_ECHO
    # TODO: 未知はコメント化
    return f"\n# TODO: 手で実装 → {L}\n"


def render_web(spec: dict) -> str:
    eps = []
    for ln in (spec.get("エンドポイント", "").splitlines() or []):
        if ln.strip():
            eps.append(ep_from_line(ln))
    if not eps:
        eps = [EP_GET_SUM, EP_POST_ECHO]
    return WEB_TMPL.format(endpoints="\n".join(eps))

# --- メイン ---

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 generator.py spec.md out_dir")
        sys.exit(1)
    spec_path = sys.argv[1]
    out_dir = Path(sys.argv[2])

    spec = parse_spec(read_file(spec_path))
    env = detect_env(spec)

    if env == "cli":
        steps = parse_steps(spec)
        code = render_cli(steps)
        write_file(out_dir / "main.py", code)
        write_file(out_dir / "requirements.txt", "")
        print("[ok] CLIコードを out/main.py に生成しました")
        print("例) echo 'Hello' | python3 out/main.py")
    else:
        code = render_web(spec)
        write_file(out_dir / "main.py", code)
        write_file(out_dir / "requirements.txt", "fastapi\nuvicorn")
        print("[ok] WebAPI を out/main.py に生成しました")
        print("起動) uvicorn main:app --reload --port 8000 --app-dir out")

if __name__ == "__main__":
    main()