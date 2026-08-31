#pip install python-osc

from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server

def mocopi_handler(address, *args):
    print(f"{address} → {args}")

dispatcher = Dispatcher()
dispatcher.map("/mocopi/*", mocopi_handler)

ip = "0.0.0.0"  # すべてのIPから受信
port = 9000

server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
print(f"Listening for mocopi OSC on port {port}...")
server.serve_forever()
