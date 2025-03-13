from flask import Flask, request, jsonify, Response, stream_with_context
from chatllm import chat_response
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

def generate_event(user_input):
    for token in chat_response(user_input, stream_to_terminal=False):
        data = json.dumps({"message":token})
        yield f"data: {data}\n\n"

    yield "event: end\n {}\n\n"

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "Please provide a message"}), 400
    
    # response = chat_response(user_input)
    # return jsonify({"response": response})

    return Response(stream_with_context(generate_event(user_input)), content_type="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)