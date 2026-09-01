import pandas as pd
import yaml

# -------------------------
# 1. Load YAML
# -------------------------
with open("gsmc_formmapping.yml", "r", encoding="utf-8") as f:
    mapping = yaml.safe_load(f)["GSMC_Form_Mapping"]

# Weighting
weights = mapping["aggregation"]["weights_by_scale"]

# -------------------------
# 2. Load CSV
# -------------------------
csv_path = "GSMCタイプ診断（回答） - フォームの回答 1.csv"
df = pd.read_csv(csv_path)

yes_value = mapping["response_file_format"]["yes_value"]
no_value  = mapping["response_file_format"]["no_value"]
name_col  = mapping["response_file_format"]["name_column"]

TYPES = ["G", "S", "M", "C"]


# =======================================================
# 3. Score calculation for each person
# =======================================================
def calc_scores(row):

    micro = {t: 0.0 for t in TYPES}
    meso  = {t: 0.0 for t in TYPES}
    macro = {t: 0.0 for t in TYPES}

    for qid, qdata in mapping["scoring"]["questions"].items():
        column = qdata["column"]
        scale  = qdata["scale"]

        answer = row[column]

        if answer == yes_value:
            score_dict = qdata.get("scores_if_yes", {})
        else:
            score_dict = qdata.get("scores_if_no", {})

        if scale == "micro":
            for t, s in score_dict.items():
                micro[t] += float(s)
        elif scale == "meso":
            for t, s in score_dict.items():
                meso[t] += float(s)
        elif scale == "macro":
            for t, s in score_dict.items():
                macro[t] += float(s)

    # -------------------------
    # 4. Weighted final score
    # -------------------------
    final = {}
    for t in TYPES:
        final[t] = (
            micro[t] * weights["micro"] +
            meso[t]  * weights["meso"] +
            macro[t] * weights["macro"]
        )

    # -------------------------
    # 5. GSMC order（順番）
    # -------------------------
    sorted_types = sorted(final.items(), key=lambda x: x[1], reverse=True)

    order_list = [t for t, _ in sorted_types]   # 例: ["C","G","S","M"]
    gsmc_order_str = "-".join(order_list)       # "C-G-S-M"
    gsmc_order_pretty = " > ".join(order_list)  # "C > G > S > M"

    # 一番最初＆一番最後
    dominant = order_list[0]
    sub      = order_list[-1]

    # -------------------------
    # 6. ratio（比率）
    # -------------------------
    total_score = sum(final.values())
    if total_score > 0:
        ratio = {t: final[t] / total_score for t in TYPES}
    else:
        ratio = {t: 0.0 for t in TYPES}

    dominant_ratio = ratio[dominant]
    sub_ratio = ratio[sub]

    return {
        "micro": micro,
        "meso": meso,
        "macro": macro,
        "final": final,
        "dominant": dominant,
        "sub": sub,
        "order_list": order_list,
        "order_str": gsmc_order_str,
        "order_str_pretty": gsmc_order_pretty,
        "ratio": ratio,
        "dominant_ratio": dominant_ratio,
        "sub_ratio": sub_ratio,
    }


# =======================================================
# 7. Execute for all rows
# =======================================================
results = []

for idx, row in df.iterrows():
    name = row[name_col]
    res = calc_scores(row)

    results.append({
        "name": name,

        # タイプ
        "dominant": res["dominant"],
        "sub": res["sub"],

        # 順番
        "order_1": res["order_list"][0],
        "order_2": res["order_list"][1],
        "order_3": res["order_list"][2],
        "order_4": res["order_list"][3],
        "GSMC_order_raw": res["order_str"],           # 例: "C-G-S-M"
        "GSMC_order_pretty": res["order_str_pretty"], # 例: "C > G > S > M",

        # スコア
        "G_score": res["final"]["G"],
        "S_score": res["final"]["S"],
        "M_score": res["final"]["M"],
        "C_score": res["final"]["C"],

        # 比率
        "G_ratio": res["ratio"]["G"],
        "S_ratio": res["ratio"]["S"],
        "M_ratio": res["ratio"]["M"],
        "C_ratio": res["ratio"]["C"],
        "dominant_ratio": res["dominant_ratio"],
        "sub_ratio": res["sub_ratio"],

        # サブスケールそのまま
        "micro_scores": res["micro"],
        "meso_scores": res["meso"],
        "macro_scores": res["macro"],
    })

results_df = pd.DataFrame(results)

# =======================================================
# 8. Save
# =======================================================
results_df.to_csv("GSMC診断_結果.csv", index=False, encoding="utf-8-sig")

print("完了！ → GSMC診断_結果.csv")
