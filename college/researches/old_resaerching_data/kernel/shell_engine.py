import re
import subprocess
import time
import os

class MiniShellEngine:
    def __init__(self):
        self.last_dir = os.getcwd()
        self.command_history = []
        self.aliases = {}

    def execute(self, command):
        command = command.strip()
        if not command:
            return ""

        # alias 展開
        if command.split()[0] in self.aliases:
            command = self.aliases[command.split()[0]] + command[len(command.split()[0]):]

        self.command_history.append(command)
        lc_command = command.lower()

        if lc_command.startswith("cd "):
            return self.handle_cd(command)
        elif lc_command == "dir":
            return self.handle_dir()
        elif lc_command.startswith("alias "):
            return self.handle_alias(command)
        elif lc_command.startswith("unalias "):
            return self.handle_unalias(command)
        elif lc_command == "history":
            return self.handle_history()
        elif lc_command == "!!":
            return self.execute(self.command_history[-2]) if len(self.command_history) > 1 else "[no previous command]"
        elif lc_command.startswith("!") and lc_command[1:].isdigit():
            idx = int(command[1:]) - 1
            if 0 <= idx < len(self.command_history):
                return self.execute(self.command_history[idx])
            else:
                return "[invalid history index]"
        elif lc_command.endswith(".py"):
            return self.handle_python(command)
        elif lc_command.endswith(".c"):
            return self.handle_c(command)
        elif lc_command.endswith(".cpp"):
            return self.handle_cpp(command)
        elif lc_command.endswith(".js"):
            return self.handle_js(command)
        else:
            return self.handle_system(command)

    def handle_cd(self, command):
        path = command[3:].strip()
        try:
            os.chdir(path)
            return f"[moved to {os.getcwd()}]"
        except FileNotFoundError:
            return "[directory not found]"

    def handle_dir(self):
        try:
            return '\n'.join(os.listdir())
        except Exception as e:
            return f"[error] {e}"

    def handle_alias(self, command):
        parts = command[6:].split('=')
        if len(parts) == 2:
            name = parts[0].strip()
            value = parts[1].strip().strip('"')
            self.aliases[name] = value
            return f"[alias set: {name}='{value}']"
        return "[invalid alias format]"

    def handle_unalias(self, command):
        name = command[8:].strip()
        if name in self.aliases:
            del self.aliases[name]
            return f"[alias removed: {name}]"
        return "[alias not found]"

    def handle_history(self):
        return '\n'.join(f"{i+1}: {cmd}" for i, cmd in enumerate(self.command_history))

    def handle_python(self, command):
        try:
            result = subprocess.run(["python", command], capture_output=True, text=True)
            return result.stdout or result.stderr
        except Exception as e:
            return f"[error] {e}"

    def handle_c(self, command):
        try:
            out = "_c.out"
            subprocess.run(["gcc", command, "-o", out], check=True)
            result = subprocess.run([f"./{out}"], capture_output=True, text=True)
            return result.stdout or result.stderr
        except Exception as e:
            return f"[error] {e}"

    def handle_cpp(self, command):
        try:
            out = "_cpp.out"
            subprocess.run(["g++", command, "-o", out], check=True)
            result = subprocess.run([f"./{out}"], capture_output=True, text=True)
            return result.stdout or result.stderr
        except Exception as e:
            return f"[error] {e}"

    def handle_js(self, command):
        try:
            result = subprocess.run(["node", command], capture_output=True, text=True)
            return result.stdout or result.stderr
        except Exception as e:
            return f"[error] {e}"

    def handle_system(self, command):
        try:
            result = subprocess.run(command.split(), capture_output=True, text=True)
            return result.stdout or result.stderr
        except Exception as e:
            return f"[error] {e}"
