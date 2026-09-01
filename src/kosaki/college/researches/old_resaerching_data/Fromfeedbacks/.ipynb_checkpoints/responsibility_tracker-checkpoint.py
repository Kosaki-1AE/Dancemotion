# responsibility_tracker.py
# 責任の矢（流入・流出）管理モジュール

class ResponsibilityTracker:
    def __init__(self):
        self.incoming_log = []
        self.outgoing_log = []

    def add_incoming(self, amount: float):
        self.incoming_log.append(amount)

    def add_outgoing(self, amount: float):
        self.outgoing_log.append(amount)

    def current_balance(self) -> float:
        if not self.incoming_log or not self.outgoing_log:
            return 0.0
        return self.outgoing_log[-1] - self.incoming_log[-1]

    def cumulative_balance(self) -> float:
        return sum(self.outgoing_log) - sum(self.incoming_log)

    def status(self) -> str:
        bal = self.current_balance()
        if bal > 0.1:
            return "Giving too much (outgoing dominant)"
        elif bal < -0.1:
            return "Receiving too much (incoming dominant)"
        else:
            return "Balanced"

if __name__ == "__main__":
    rt = ResponsibilityTracker()
    rt.add_incoming(0.3)
    rt.add_outgoing(0.5)
    print("Current:", rt.current_balance())
    print("Status:", rt.status())
