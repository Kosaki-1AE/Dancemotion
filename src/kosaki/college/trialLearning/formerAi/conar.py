import serial
import requests
import socket

def get_ip_address():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

print("IPアドレス:", get_ip_address())


# 送信先のウェブサイトのURLを指定します
url = 'https://172.20.10.11/gps.html'

# VSCodeで開いているファイルの内容を読み込みます
with open('path/to/your/code/file.py', 'r') as file:
    code_content = file.read()

# リクエストボディを作成します
data = {'code': code_content}

# POSTリクエストを送信します
response = requests.post(url, data=data)

# レスポンスを表示します
print(response.text)


# Arduinoのシリアルポートを設定
ser = serial.Serial('COM15', 115200)  # ポート名とボーレートをArduinoに合わせて設定

while True:
    # Arduinoからデータを読み取る
    data = ser.readline().decode().strip()
    print(f"M5stackからのデータ: {data}")

# シリアルポートを閉じる
ser.close()
