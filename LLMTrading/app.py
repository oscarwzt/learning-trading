from flask import Flask, render_template, request, Response
import openai
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama

app = Flask(__name__)

# Read your API key (if needed for OpenAI services)
with open("api.txt", "r") as file:
    openai.api_key = file.read().strip()

# Load the storage context
storageContext = StorageContext.from_defaults(
    persist_dir="tradingVolatilityDB")
# Load the index
index = load_index_from_storage(storageContext)

# Create Ollama LLM (adjust parameters as needed)
deepseek = Ollama(model="llama3.2", request_timeout=120, max_tokens=512)

# Create the chat engine
chat_engine = index.as_chat_engine(
    llm=deepseek,
    similarity_top_k=5,
    use_async=False,
    chat_mode="context",
    streaming=True
)


@app.route("/")
def index():
    """
    Render the main chat page.
    """
    return render_template("index.html")


@app.route("/stream", methods=["POST"])
def stream():
    """
    SSE endpoint: returns partial responses (stream) from the LLM.
    This route is invoked by JavaScript on the frontend.
    """
    user_message = request.form.get("user_message", "")

    # Define a generator function that yields partial LLM response chunks.
    def generate():
        # Here we do a *conceptual* streaming approach.
        # LlamaIndex + Ollama might not support partial chunk streaming by default.
        # Adjust or use your own method to retrieve chunked text if available.

        # We'll just do a final response as a single chunk for simplicity,
        # but you'd replace this with your streaming approach if supported.
        response_obj = chat_engine.chat(user_message)
        full_text = response_obj.response

        # Example chunking: yield text in small pieces
        chunk_size = 40
        for i in range(0, len(full_text), chunk_size):
            yield f"data: {full_text[i:i+chunk_size]}\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
