# main_qiskit_demo.py
# -------------------
# qstillness_engine を単体実行して画像保存。
# pip install qiskit matplotlib numpy

import matplotlib.pyplot as plt
import numpy as np

from stillness_lightning_qiskit import (SimParams,
                                        run_qiskit_stillness_lightning)


def main():
    params = SimParams(T=300, tension_threshold=0.95)
    log = run_qiskit_stillness_lightning(params)

    t = np.arange(len(log.p_motion))
    plt.figure(figsize=(10,4))
    plt.plot(t, log.p_motion, label="P(Motion)")
    plt.plot(t, log.tension,  label="Tension")
    plt.plot(t, log.fear,     label="Fear")
    plt.plot(t, log.wonder,   label="Wonder")
    plt.legend(); plt.tight_layout()
    plt.savefig("qiskit_stillness_lightning.png")
    print("Saved: qiskit_stillness_lightning.png")
    print("Events:", log.events[:8])
    print("Memory:", log.memory[:3])

if __name__ == "__main__":
    main()
