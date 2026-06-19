from flask import Flask, request, jsonify
from pathlib import Path
import importlib.util

# Dynamically load the tools from the filename 08.mcp.py
tools_path = Path(__file__).parent / "08.mcp.py"
spec = importlib.util.spec_from_file_location("mcp_tools", str(tools_path))
mcp_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_tools)

app = Flask(__name__)


@app.route("/")
def index():
    return jsonify({
        "endpoints": {
            "/echo (POST)": {"body": {"text": "string"}},
            "/add (POST)": {"body": {"a": "number", "b": "number"}},
            "/datetime (GET)": {}
        }
    })


@app.route("/echo", methods=["POST"])
def echo_route():
    data = request.get_json(silent=True) or {}
    text = data.get("text") or data.get("input") or ""
    return jsonify({"result": mcp_tools.echo(text)})


@app.route("/add", methods=["POST"])
def add_route():
    data = request.get_json(silent=True) or {}
    try:
        a = float(data.get("a", 0))
        b = float(data.get("b", 0))
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"result": mcp_tools.add_numbers(a, b)})


@app.route("/datetime", methods=["GET"])
def datetime_route():
    return jsonify({"result": mcp_tools.get_datetime()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
