from flask import Flask, request, render_template
from shell_engine import MiniShellEngine

app = Flask(__name__)
shell = MiniShellEngine()

@app.route('/', methods=['GET', 'POST'])
def index():
    output = ""
    if request.method == 'POST':
        command = request.form.get('command', '')
        if command:
            output = shell.execute(command)
    return render_template('index.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)
