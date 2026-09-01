import re
import subprocess
import time
import os


class MiniShellEngine:
    def __init__(self):
        self.last_dir = None
        self.command_history = []
        self.aliases = {}
        self.commands = {}
        self.RC_FILE = ".selfrc"
        self.HISTORY_FILE = ".history.txt"
        self.register_builtin_commands()
        self.load_aliases()
        self.load_history()

    def register_command(self, name, func):
        self.commands[name] = func

    def run(self):
        print("🧠 Hello World! ")
        while True:
            try:
                command = input(self.get_prompt()).strip()
                if not command:
                    continue
                elif not self.execute_command(command):
                    try:
                        subprocess.run(command, shell=True)
                    except FileNotFoundError:
                        print("Command not found")
            except KeyboardInterrupt:
                print("\n[INTERRUPTED]")
                break

    def execute_command(self, command):
        command = command.strip()
        if command:
            self.command_history.append(command)
            self.save_history()

        tokens = command.split()
        if not tokens:
            return False

        # alias 展開
        if tokens[0] in self.aliases:
            command = self.aliases[tokens[0]] + ' ' + ' '.join(tokens[1:])
            tokens = command.split()

        cmd = tokens[0].lower()

        if cmd in self.commands:
            return self.commands[cmd](command)
        elif cmd == "!!":
            if len(self.command_history) > 1:
                last_cmd = self.command_history[-2]
                print(f"→ {last_cmd}")
                return self.execute_command(last_cmd)
        elif cmd.startswith("!") and cmd[1:].isdigit():
            index = int(cmd[1:]) - 1
            if 0 <= index < len(self.command_history):
                print(f"→ {self.command_history[index]}")
                return self.execute_command(self.command_history[index])
        elif cmd.startswith("!"):
            keyword = cmd[1:]
            for past_cmd in reversed(self.command_history):
                if past_cmd.startswith(keyword):
                    print(f"→ {past_cmd}")
                    return self.execute_command(past_cmd)
        elif cmd.endswith(".py"):
            return self.handle_python_exec(command)
        elif cmd.endswith(".c"):
            return self.handle_c_exec(command)
        elif cmd.endswith(".cpp"):
            return self.handle_cpp_exec(command)
        elif cmd.endswith(".js"):
            return self.handle_js_exec(command)
        elif cmd == "exit":
            print("Bye.")
            exit()

        return False

    def get_prompt(self):
        if self.last_dir:
            try:
                current = os.getcwd()
                if current != self.last_dir:
                    rel = os.path.relpath(current, self.last_dir)
                    if rel != ".":
                        return f"[{rel}] → "
                return "→ "
            except Exception:
                return "→ "
        return "→ "

    def save_aliases(self):
        with open(self.RC_FILE, 'w') as f:
            for k, v in self.aliases.items():
                f.write(f"alias {k}='{v}'\n")

    def load_aliases(self):
        if os.path.exists(self.RC_FILE):
            with open(self.RC_FILE) as f:
                for line in f:
                    if line.startswith("alias"):
                        self.handle_alias(line.strip())

    def save_history(self):
        with open(self.HISTORY_FILE, 'w') as f:
            f.write("\n".join(self.command_history))

    def load_history(self):
        if os.path.exists(self.HISTORY_FILE):
            with open(self.HISTORY_FILE) as f:
                for line in f:
                    cmd = line.strip()
                    if cmd:
                        self.command_history.append(cmd)

    def register_builtin_commands(self):
        self.register_command("cd", self.handle_cd)
        self.register_command("dir", self.handle_dir)
        self.register_command("show", self.handle_show)
        self.register_command("wait", self.handle_wait)
        self.register_command("change", self.handle_change)
        self.register_command("clear", self.handle_clear)
        self.register_command("history", self.handle_history)
        self.register_command("mkdir", self.handle_mkdir)
        self.register_command("touch", self.handle_touch)
        self.register_command("history -c", self.handle_history_clear)
        self.register_command("alias", self.handle_alias)
        self.register_command("unalias", self.handle_unalias)
        self.register_command("echo", self.handle_echo)
        self.register_command("rm", self.handle_rm)
        self.register_command("cat", self.handle_cat)

    def handle_cd(self, command):
        path = command[3:].strip()
        try:
            os.chdir(path)
        except FileNotFoundError:
            pass
        return True

    def handle_dir(self, *_):
        self.last_dir = os.getcwd()
        for item in os.listdir():
            print(item)
        return True

    def handle_show(self, command):
        nums = re.findall(r"\d+", command)
        for num in nums:
            print(num)
            time.sleep(0.3)
        return True

    def handle_wait(self, *_):
        time.sleep(0.5)
        return True

    def handle_change(self, command):
        nums = re.findall(r"\d+", command)
        for num in nums:
            print(num)
            time.sleep(0.3)
        return True

    def handle_clear(self, *_):
        print("\033[2J\033[H", end='')
        return True

    def handle_history(self, *_):
        for i, cmd in enumerate(self.command_history, 1):
            print(f"{i}: {cmd}")
        return True

    def handle_history_clear(self, *_):
        self.command_history.clear()
        self.save_history()
        print("[history cleared]")
        return True

    def handle_mkdir(self, command):
        folder_name = command[6:].strip()
        if folder_name:
            os.makedirs(folder_name, exist_ok=True)
        return True

    def handle_touch(self, command):
        file_name = command[6:].strip()
        if file_name:
            open(file_name, 'a').close()
        return True

    def handle_alias(self, command):
        parts = command.split("=", 1)
        if len(parts) == 2:
            key = parts[0].strip().split()[1]
            value = parts[1].strip().strip("'").strip('"')
            self.aliases[key] = value
            print(f"[alias set] {key}='{value}'")
            self.save_aliases()
        elif command.strip() == "alias":
            for k, v in self.aliases.items():
                print(f"{k}='{v}'")
        return True

    def handle_unalias(self, command):
        parts = command.strip().split()
        if len(parts) == 2 and parts[1] in self.aliases:
            del self.aliases[parts[1]]
            print(f"[alias removed] {parts[1]}")
            self.save_aliases()
        return True

    def handle_echo(self, command):
        print(command[5:].strip())
        return True

    def handle_rm(self, command):
        path = command[3:].strip()
        if os.path.exists(path):
            os.remove(path)
            print(f"[removed] {path}")
        return True

    def handle_cat(self, command):
        path = command[4:].strip()
        if os.path.exists(path):
            with open(path) as f:
                print(f.read())
        return True

    def handle_python_exec(self, command):
        try:
            subprocess.run(["python", command])
        except Exception as e:
            print(f"[error] {e}")
        return True

    def handle_c_exec(self, command):
        try:
            exe_name = command.replace(".c", "_c.out")
            subprocess.run(["gcc", command, "-o", exe_name])
            subprocess.run([f"./{exe_name}"])
        except Exception as e:
            print(f"[error] {e}")
        return True

    def handle_cpp_exec(self, command):
        try:
            exe_name = command.replace(".cpp", "_cpp.out")
            subprocess.run(["g++", command, "-o", exe_name])
            subprocess.run([f"./{exe_name}"])
        except Exception as e:
            print(f"[error] {e}")
        return True

    def handle_js_exec(self, command):
        try:
            subprocess.run(["node", command])
        except Exception as e:
            print(f"[error] {e}")
        return True


if __name__ == "__main__":
    shell = MiniShellEngine()
    shell.run()
