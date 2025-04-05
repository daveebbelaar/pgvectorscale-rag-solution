from flask import Flask, request, jsonify
from datetime import datetime
from app.database.vector_store import VectorStore
from app.services.synthesizer import Synthesizer

app = Flask(__name__)

vec = VectorStore()

@app.route('/api/chat', methods=['POST'])
def chat():
    question = request.json['question']
    results = vec.search(question, limit=30)
    response = Synthesizer.generate_response(question=question, context=results)
    return jsonify({'answer': response.answer})

if __name__ == '__main__':
    app.run(debug=True)
