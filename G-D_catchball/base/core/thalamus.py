from __future__ import annotations

import numpy as np

from base.core.BasalGangliaAmygdala import CellSystem
from data.merge.multimodal_integrator import FusionResult, MultiModalIntegrator


class MultiModalToCellBridge:
    """
    multimodal_integrator の fused_vector を
    CellSystem の semantic.vector に直接流し込む接続クラス。
    """

    def __init__(self, system: CellSystem, integrator: MultiModalIntegrator):
        self.system = system
        self.integrator = integrator

    def step(self, label: str = "multimodal_input"):
        # 1. word/music/video のCSVを読んで統合
        evidences, fusion_result = self.integrator.run(export_prefix="fusion")

        # 2. 統合ベクトルを取り出す
        fused_vector = np.asarray(fusion_result.fused_vector, dtype=float)

        # 3. CellSystemへ入力
        self.system.bus["text"] = label
        self.system.bus["semantic.vector"] = fused_vector
        self.system.bus["novelty"] = fusion_result.novelty_score
        self.system.bus["conflict"] = fusion_result.conflict_score
        self.system.bus["agreement"] = fusion_result.agreement_score

        # 4. 自我・ズレ・責任の矢を更新
        self.system.run()

        # 5. 結果をまとめて返す
        return {
            "fused_vector": fused_vector,
            "fusion_result": fusion_result,
            "evidences": evidences,
            "cell_bus": self.system.bus,
            "state": self.system.bus.get("state", {}),
            "self": self.system.bus.get("self", np.zeros(3)),
            "score": self.system.bus.get("score", 0.0),
            "deviation": self.system.bus.get("deviation", {}),
            "responsibility_arrow": self.system.bus.get("responsibility_arrow", np.zeros(3)),
        }