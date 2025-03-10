import uvicorn
import argparse
import os
import signal
import sys
import time
import platform
import subprocess
from datetime import datetime

# Configurable settings
HOST = "127.0.0.1"  # Change to "0.0.0.0" for external access
PORT = 8000
APP_MODULE = "chatbot:app"  # Format: "module_name:fastapi_instance"
PID_FILE = "uvicorn.pid"  # File to store the process ID


def start_server():
    """Starts the Uvicorn server without auto-reload for better performance."""
    if os.path.exists(PID_FILE):
        print("⚠️ Server is already running! Stop it first before starting again.")
        sys.exit(1)

    start_time = time.time()  # ⏳ Record start time

    print(f"🚀 Starting Uvicorn server on {HOST}:{PORT} (Debug: ON, Auto-reload: OFF)")

    # Start Uvicorn as a subprocess (NO --reload for better performance)
    process = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        APP_MODULE,
        "--host", HOST,
        "--port", str(PORT),
        "--log-level", "debug"  # Debug mode enabled
    ], stdout=sys.stdout, stderr=sys.stderr)

    # Save the PID so we can stop the server later
    with open(PID_FILE, "w") as f:
        f.write(str(process.pid))

    # Wait for Uvicorn to fully start
    time.sleep(2)  # ⏳ Give time for the server to start

    elapsed_time = time.time() - start_time  # ⏳ Calculate startup time
    startup_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 📅 Timestamp

    print(f"✅ Server started at {startup_timestamp} (PID: {process.pid})")
    print(f"⏳ Server took {elapsed_time:.2f} seconds to start.")


def stop_server():
    """Stops the Uvicorn server and ensures the PID file is deleted."""
    if not os.path.exists(PID_FILE):
        print("⚠️ No running server found.")
        return

    with open(PID_FILE, "r") as f:
        pid = int(f.read().strip())

    print(f"🛑 Stopping Uvicorn server (PID: {pid})...")

    system_name = platform.system()

    if system_name == "Windows":
        # Kill only the specific Uvicorn process, NOT all Python processes
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif system_name in ["Linux", "Darwin"]:
        try:
            os.kill(pid, signal.SIGTERM)  # Graceful shutdown
        except ProcessLookupError:
            print(f"⚠️ Process {pid} already terminated.")

    # Wait to ensure the process is stopped
    time.sleep(2)

    # ✅ Ensure PID file is deleted
    if os.path.exists(PID_FILE):
        try:
            os.remove(PID_FILE)
            print("✅ Server stopped successfully. PID file removed.")
        except Exception as e:
            print(f"⚠️ Failed to delete PID file: {e}")
    else:
        print("✅ Server stopped successfully. PID file was already removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage the Uvicorn server.")
    parser.add_argument("action", choices=["start", "stop"], help="Start or stop the Uvicorn server.")

    args = parser.parse_args()

    if args.action == "start":
        start_server()
    elif args.action == "stop":
        stop_server()

#python server_manager.py start
#http://localhost:8000/static/index.html
#python server_manager.py stop