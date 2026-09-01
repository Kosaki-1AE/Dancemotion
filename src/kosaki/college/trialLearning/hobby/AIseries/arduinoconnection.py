from flask import Flask
import socket

def get_current_ip_address():
    try:
        # ホスト名を取得
        hostname = socket.gethostname()
        
        # ホスト名からIPアドレスを解決
        ip_address = socket.gethostbyname(hostname)
        
        return ip_address
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    current_ip = get_current_ip_address()
    if current_ip:
        print(f"現在のIPアドレスは {current_ip} です。")
    else:
        print("IPアドレスを取得できませんでした。")


app = Flask(__name__)

@app.route('/your_endpoint', methods=['GET'])
def handle_request():
    # Arduinoからのリクエストを処理
    # 必要な処理を記述
    return "Hello from Python!"

if __name__ == '__main__':
    app.run(host=current_ip, port=80)