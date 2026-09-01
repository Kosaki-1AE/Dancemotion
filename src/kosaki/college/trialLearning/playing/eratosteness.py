from flask import Flask, render_template
app = Flask(__name__)
@app.route('/')
def index():
    message = "素数リスト",prime
    num = int(input("整数を入力："))
    prime = []
    for i in range(2, num+1):
        is_prime = True
        for j in range(2, int(i ** 0.5) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            prime.append(i)
    return render_template('index.html',message=message)
if __name__ == '__main__':
    app.run()