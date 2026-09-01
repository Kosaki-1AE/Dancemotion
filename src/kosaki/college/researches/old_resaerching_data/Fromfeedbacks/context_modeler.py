# context_modeler.py
# 文脈や状況ごとの構造知変化を記録・切り替えられるモデル

class ContextModeler:
    def __init__(self):
        self.contexts = {}
        self.current_context = None

    def create_context(self, name: str):
        if name not in self.contexts:
            self.contexts[name] = {
                "emotion_log": [],
                "airflow_log": [],
                "responsibility_log": [],
                "structure_log": []
            }
            self.current_context = name

    def switch_context(self, name: str):
        if name in self.contexts:
            self.current_context = name
        else:
            raise ValueError(f"Context '{name}' does not exist.")

    def log_state(self, emotion, airflow, responsibility, structure):
        if not self.current_context:
            raise RuntimeError("No context selected.")
        self.contexts[self.current_context]["emotion_log"].append(emotion)
        self.contexts[self.current_context]["airflow_log"].append(airflow)
        self.contexts[self.current_context]["responsibility_log"].append(responsibility)
        self.contexts[self.current_context]["structure_log"].append(structure)

    def get_context_data(self, name: str):
        return self.contexts.get(name, {})

    def list_contexts(self):
        return list(self.contexts.keys())

if __name__ == "__main__":
    cm = ContextModeler()
    cm.create_context("dance")
    cm.log_state("joy", 0.3, -0.1, [0.6, 0.4, 0.3])
    cm.create_context("talk")
    cm.switch_context("talk")
    cm.log_state("trust", 0.5, 0.1, [0.7, 0.6, 0.5])
    print("Contexts:", cm.list_contexts())
    print("Dance log:", cm.get_context_data("dance"))
