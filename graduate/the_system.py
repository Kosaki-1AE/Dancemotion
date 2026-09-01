# =========================
# Field-Based System (Psychological Choice UI + Oblate Layer)
# genesis -> buffer -> motion -> coherence -> capacity -> quality -> stillness
# =========================

import copy


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


# -------------------------
# 設定
# -------------------------
visible_fields = [
    "buffer",
    "motion",
    "coherence",
    "capacity",
    "quality",
    "stillness",
]

show_genesis_summary = True
show_current_state_summary = True
show_depth_core_summary = True
wanted_top_n = 4


# -------------------------
# State
# -------------------------
def init_state():
    return {
        "buffer": 0.70,
        "depth": 0.20,
    }


# -------------------------
# Input -> Signal
# input は命令ではなく genesis 側の刺激
# -------------------------
def interpret_input(user_input: str):
    signal = {
        "expand": 0.0,
        "settle": 0.0,
        "open": 0.0,
        "focus": 0.0,
        "text": user_input,
    }

    if "揺" in user_input:
        signal["expand"] += 0.35
    if "落ち着" in user_input:
        signal["settle"] += 0.30
    if "解放" in user_input:
        signal["open"] += 0.35
    if "深" in user_input or "核" in user_input:
        signal["focus"] += 0.35

    if sum(v for k, v in signal.items() if k != "text") == 0:
        signal["settle"] += 0.05

    return signal


# -------------------------
# Genesis
# -------------------------
def generate_genesis(state, signal):
    genesis = []

    genesis.append({
        "name": "gen_A",
        "db": 0.05 + 0.10 * state["buffer"] + 0.18 * signal["settle"],
        "dd": 0.01 + 0.04 * state["depth"],
        "flow": 0.03,
        "trace": "settle-rise"
    })

    genesis.append({
        "name": "gen_B",
        "db": -0.02 + 0.02 * signal["open"],
        "dd": 0.01 + 0.03 * signal["expand"],
        "flow": 0.08 + 0.18 * signal["expand"],
        "trace": "expand-rise"
    })

    genesis.append({
        "name": "gen_C",
        "db": 0.02 + 0.06 * signal["open"],
        "dd": 0.005 + 0.02 * state["depth"],
        "flow": 0.02,
        "trace": "release-rise"
    })

    genesis.append({
        "name": "gen_D",
        "db": 0.01 + 0.04 * signal["focus"],
        "dd": 0.03 + 0.18 * state["depth"] + 0.20 * signal["focus"],
        "flow": 0.01,
        "trace": "deepen-rise"
    })

    for g in genesis:
        g["db"] = round(g["db"], 4)
        g["dd"] = round(g["dd"], 4)
        g["flow"] = round(g["flow"], 4)

    return genesis


# -------------------------
# 評価軸
# -------------------------
def calc_buffer_score(state, g):
    score = state["buffer"]
    if g["db"] < 0:
        score -= abs(g["db"]) * 2.0
    return clamp(round(score, 4))


def calc_motion_score(state, g):
    score = g["flow"] + max(0.0, g["dd"]) * 0.5
    return clamp(round(score, 4))


def calc_coherence_score(state, g):
    score = 0.0
    score += state["buffer"] * 0.35
    score += state["depth"] * 0.35
    score += max(0.0, g["dd"]) * 1.5
    score += max(0.0, g["db"]) * 0.5
    score += g["flow"] * 0.7
    score -= max(0.0, -g["db"]) * 0.8
    return clamp(round(score, 4))


def calc_capacity_score(state, g):
    load = abs(g["db"]) + abs(g["dd"]) + g["flow"]
    cap = state["buffer"] + state["depth"] * 0.5 - load * 0.7
    return clamp(round(cap, 4))


def calc_quality_score(state, g, coherence, capacity):
    score = coherence * 0.6 + capacity * 0.4
    if g["dd"] > 0:
        score += 0.08
    return clamp(round(score, 4))


def calc_stillness_score(state, g):
    score = state["buffer"] * 0.7 + max(0.0, 0.15 - g["flow"]) * 1.2
    return clamp(round(score, 4))


# -------------------------
# 候補構築
# -------------------------
def build_transition_candidates(state, genesis_list):
    candidates = []

    for i, g in enumerate(genesis_list, start=1):
        buffer_score = calc_buffer_score(state, g)
        motion_score = calc_motion_score(state, g)
        coherence_score = calc_coherence_score(state, g)
        capacity_score = calc_capacity_score(state, g)
        quality_score = calc_quality_score(state, g, coherence_score, capacity_score)
        stillness_score = calc_stillness_score(state, g)

        candidate = copy.deepcopy(g)
        candidate["id"] = i
        candidate["buffer"] = buffer_score
        candidate["motion"] = motion_score
        candidate["coherence"] = coherence_score
        candidate["capacity"] = capacity_score
        candidate["quality"] = quality_score
        candidate["stillness"] = stillness_score
        candidate["label"] = emergent_label(candidate)
        candidates.append(candidate)

    candidates.sort(key=lambda x: (x["quality"], x["coherence"]), reverse=True)

    for i, c in enumerate(candidates, start=1):
        c["id"] = i

    return candidates


# -------------------------
# ラベル
# -------------------------
def emergent_label(c):
    if c["dd"] > 0.08:
        return "deepen-like"
    if c["db"] > 0.08 and c["flow"] < 0.05:
        return "stabilize-like"
    if c["flow"] > 0.12:
        return "perturb-like"
    if c["db"] > 0 and c["flow"] < 0.04:
        return "release-like"
    return "stay-like"


# -------------------------
# オブラート層推定
# -------------------------
def infer_oblate_layer(c):
    motion = c["motion"]
    stillness = c["stillness"]
    coherence = c["coherence"]
    capacity = c["capacity"]
    quality = c["quality"]
    label = c["label"]

    # まず静に近いところを先判定
    if stillness >= 0.72 and motion <= 0.03:
        return "オブラート5枚目", "静"

    if stillness >= 0.67 and quality >= 0.62:
        return "オブラート4.5枚目", "狂/静"

    if label == "deepen-like" and quality >= 0.64:
        return "オブラート4枚目", "得/狂"

    if label == "stabilize-like" and coherence >= 0.42 and capacity >= 0.66:
        return "オブラート3.5枚目", "普/得"

    if label == "release-like" and stillness > 0.62:
        return "オブラート3枚目", "得/鬱"

    if 0.38 <= coherence < 0.48 and stillness > 0.60:
        return "オブラート2.5枚目", "普/得"

    if coherence < 0.38 and capacity > 0.68:
        return "オブラート2枚目", "楽/普"

    if motion > 0.10 and quality < 0.58:
        return "オブラート1.5枚目", "楽"

    if motion > 0.14:
        return "オブラート1枚目", "動"

    return "オブラート2枚目", "普"


# -------------------------
# 要約
# -------------------------
def summarize_genesis(signal, candidates):
    if not candidates:
        return "Genesis: 候補なし"

    top = candidates[0]
    parts = []

    if signal["focus"] > 0:
        parts.append("深まり寄り")
    if signal["expand"] > 0:
        parts.append("拡張寄り")
    if signal["settle"] > 0:
        parts.append("安定寄り")
    if signal["open"] > 0:
        parts.append("解放寄り")
    if not parts:
        parts.append("微弱な待機寄り")

    return f"Genesis: {' / '.join(parts)} | 先頭候補={top['label']} ({top['trace']})"


def summarize_state(state):
    return f"State: buffer={state['buffer']:.3f}, depth={state['depth']:.3f}"


def split_candidates(candidates):
    wanted = candidates[:wanted_top_n]
    shadow = candidates[wanted_top_n:]
    return wanted, shadow


def summarize_depth_core(state, candidates):
    top = candidates[:wanted_top_n]

    if not top:
        return "Depth Core: まだ核が立っていない"

    labels = [c["label"] for c in top]
    avg_quality = sum(c["quality"] for c in top) / len(top)
    avg_stillness = sum(c["stillness"] for c in top) / len(top)
    avg_motion = sum(c["motion"] for c in top) / len(top)

    if any("deepen" in x for x in labels) and any("stabilize" in x for x in labels):
        core = "安定しつつ仲良くなりたい"
    elif any("perturb" in x for x in labels):
        core = "覚悟してでも動かなきゃ"
    elif any("release" in x for x in labels):
        core = "力抜いてもいいかなぁ"
    else:
        core = "まだちょっと待って。"

    mode = []
    if avg_quality > 0.62:
        mode.append("求心強め")
    if avg_stillness > 0.62:
        mode.append("静けさ高め")
    if avg_motion > 0.10:
        mode.append("遷移欲あり")

    suffix = f" / {', '.join(mode)}" if mode else ""
    return f"Depth Core: 今は『{core}』が核{suffix}"


# -------------------------
# 心理描写
# -------------------------
def candidate_psychology(c):
    label = c["label"]
    motion = c["motion"]
    stillness = c["stillness"]
    quality = c["quality"]
    coherence = c["coherence"]
    capacity = c["capacity"]

    oblate_layer, oblate_mode = infer_oblate_layer(c)

    if label == "release-like" and stillness > 0.62:
        tone = "ツンデレ"
    elif label == "perturb-like" and motion > 0.12:
        tone = "衝動"
    elif label == "deepen-like" and quality > 0.65:
        tone = "沈潜"
    elif motion > 0.10 and coherence < 0.45:
        tone = "焦り"
    elif stillness > 0.64 and capacity > 0.65:
        tone = "素直"
    elif capacity > 0.70 and quality < 0.62:
        tone = "拗ね"
    else:
        tone = "観測"

    if label == "stabilize-like":
        want = "安定しつつ仲良くなりたい"
        defense = "心の距離、このままでいいかも。"
        monologue = "まだ自分と相談してたい。急に踏み込むとかは嫌。息合ってないもん。"

    elif label == "release-like":
        want = "力抜いてもいいかなぁ"
        defense = "直球で晒したくない"
        monologue = "ほんとは通したい。でもそのまま見せるのはちょっと癪。だから一回ゆるめて近づきたいな...。"

    elif label == "deepen-like":
        want = "もっと奥まで知りたい"
        defense = "浅いまま終わるのは嫌"
        monologue = "表面で済ませたくはない。ちゃんと芯まで触れてみたい。"

    elif label == "perturb-like":
        want = "覚悟してでも動かなきゃ"
        defense = "停滞したまま閉じるのは絶対避けたい"
        monologue = "このまま収まるのは絶対に違う。多少荒れても良いから、自分の中の何かを動かしたい。"

    else:
        want = "まだちょっと待って。"
        defense = "今ここで確定しすぎるのは避けたい"
        monologue = "まだ決めきらない。少し距離を取って様子を見たい。"

    if tone == "ツンデレ":
        monologue += " ただ、素直に欲しいとは言いたくない(恥ずかしいし...)。"
    elif tone == "拗ね":
        monologue += " うまく行けそうなのに、わざと別ルートを見てるのがすっごいうずうずする。"
    elif tone == "焦り":
        monologue += " じっとしていると置いていかれそうでなんか落ち着かない。"
    elif tone == "沈潜":
        monologue += " 今は表面とかよりも、内面を見てみたいかも。"

    return {
        "tone": tone,
        "want": want,
        "defense": defense,
        "monologue": monologue,
        "oblate_layer": oblate_layer,
        "oblate_mode": oblate_mode,
    }


# -------------------------
# 表示
# -------------------------
def format_candidate_block(c):
    psy = candidate_psychology(c)

    lines = []
    lines.append(
        f"[{c['id']}] {c['label']} / tone: {psy['tone']} / {psy['oblate_layer']} ({psy['oblate_mode']})"
    )
    lines.append(f"  {psy['monologue']}")
    lines.append(f"  want: {psy['want']}")
    lines.append(f"  defense: {psy['defense']}")
    lines.append(f"  trace: {c['trace']}")

    metrics = []
    for name in visible_fields:
        if name in c:
            metrics.append(f"{name}={c[name]:.3f}")

    lines.append("  " + " | ".join(metrics))
    return "\n".join(lines)


def print_candidates_grouped(candidates):
    wanted, shadow = split_candidates(candidates)

    print("Wanted transitions:")
    for c in wanted:
        print(format_candidate_block(c))
        print()

    if shadow:
        print("Shadow / Twisted transitions:")
        for c in shadow:
            print(format_candidate_block(c))
            print()


# -------------------------
# 適用
# -------------------------
def apply_candidate(state, c):
    new_state = copy.deepcopy(state)

    new_state["buffer"] += c["db"]
    new_state["depth"] += c["dd"] + 0.20 * c["motion"]
    new_state["depth"] += 0.05 * c["quality"]

    new_state["buffer"] = clamp(round(new_state["buffer"], 6))
    new_state["depth"] = clamp(round(new_state["depth"], 6))

    return new_state


def calc_responsibility(before, after, c):
    db = after["buffer"] - before["buffer"]
    dd = after["depth"] - before["depth"]
    delta = abs(db) + abs(dd)
    return round(delta * (1.0 + c["coherence"]), 4)


# -------------------------
# ターン準備
# -------------------------
def prepare_turn(state, user_input):
    signal = interpret_input(user_input)
    genesis_list = generate_genesis(state, signal)
    candidates = build_transition_candidates(state, genesis_list)
    return signal, candidates


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    state = init_state()

    print("=== Field-Based Psychological Choice System ===")
    print("genesis刺激から候補遷移を立ち上げ、Depth Core と心理描写つき選択肢を表示します。")
    print("表示軸:", ", ".join(visible_fields))
    print("終了: exit")
    print("-" * 60)

    while True:
        if show_current_state_summary:
            print(summarize_state(state))

        user_input = input("genesis >> ").strip()

        if user_input == "exit":
            break

        signal, candidates = prepare_turn(state, user_input)

        if show_depth_core_summary:
            print(summarize_depth_core(state, candidates))

        if show_genesis_summary:
            print(summarize_genesis(signal, candidates))

        print_candidates_grouped(candidates)

        choice = input("select >> ").strip()

        if choice == "exit":
            break

        if not choice.isdigit():
            print("番号で選んでくれ")
            print("-" * 60)
            continue

        choice_num = int(choice)
        selected = next((c for c in candidates if c["id"] == choice_num), None)

        if selected is None:
            print("その番号はない")
            print("-" * 60)
            continue

        before = copy.deepcopy(state)
        state = apply_candidate(state, selected)
        responsibility = calc_responsibility(before, state, selected)

        psy = candidate_psychology(selected)

        print(
            f"selected: {selected['label']} / tone: {psy['tone']} / {psy['oblate_layer']} ({psy['oblate_mode']})"
        )
        print(f"inner voice: {psy['monologue']}")
        print(f"responsibility: {responsibility:.4f}")
        print(f"state: buffer={state['buffer']:.4f}, depth={state['depth']:.4f}")
        print("-" * 60)