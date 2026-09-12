"""
ARIYATHE THALAYATTIYATH — Local Python Launcher
Launches local web server and automatically opens the project in your browser.
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def run():
    os.chdir(DIRECTORY)
    
    # Try port 8000, fallback to other ports if busy
    global PORT
    server = None
    for attempt_port in range(PORT, PORT + 10):
        try:
            server = socketserver.TCPServer(("", attempt_port), Handler)
            PORT = attempt_port
            break
        except OSError:
            continue
            
    if not server:
        print("[-] Could not bind to port 8000-8010. Please close conflicting programs.")
        sys.exit(1)

    url = f"http://localhost:{PORT}"
    print("=" * 65)
    print("⚡ ARIYATHE THALAYATTIYATH — AI UNDERSTANDING DETECTION SYSTEM")
    print(f"📡 Server running at: {url}")
    print("🚀 Opening your browser automatically...")
    print("💡 Press Ctrl+C in this terminal window to stop the server.")
    print("=" * 65)

    # Open default browser
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped. Have a great hackathon presentation!")
        server.server_close()

if __name__ == '__main__':
    run()
