import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app) 

# --- API KEYS ---
PINECONE_API_KEY = "pcsk_5q84SC_N75kyYFFVJFtYxFNNTDuFDiFC6cWTnzzPyCHx1pUtMg3Zb3zoPFvGwPKPerMAyr"
GEMINI_API_KEY = "AIzaSyAOhOiFQ8AhqmF7FUvgLSkpprrQJ4msVjE"

print("Starting PlanA Backend with Gemini...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
pc = Pinecone(api_key=PINECONE_API_KEY)                   
index = pc.Index("plana-ai-db")
ai_client = genai.Client(api_key=GEMINI_API_KEY)

@app.route('/ask', methods=['POST'])
def ask_ai():
    data = request.json
    student_question = data.get('question', '')
    base64_image = data.get('image')

    try:
        # 1. Search Pinecone Database
        query_vector = embedding_model.encode(student_question).tolist()
        results = index.query(vector=query_vector, top_k=1, include_metadata=True)
        best_match = results['matches'][0]['metadata']['text']
        
        # 2. Build the Strict Prompt
        prompt = f"""
        អ្នកគឺជាគ្រូបង្រៀនគណិតវិទ្យាដ៏ពូកែម្នាក់នៅកម្ពុជា តំណាងឱ្យស្ថាប័ន PlanA Ai។
        
        TASK: Solve the math problem asked by the user. If they attached an image, read the math from the image.
        
        STRICT RULES:
        1. Output your step-by-step explanation EXCLUSIVELY in the Khmer language.
        2. Use standard high school methods. NEVER use advanced calculus (like derivatives) for limits or asymptotes unless explicitly asked.
        3. Use standard LaTeX formatting for math.
        4. Use the 'Reference Data' below to guide your formatting and method IF it is relevant. If it is a completely different topic, ignore it.
        
        REFERENCE DATA (from database):
        {best_match}
        
        USER QUESTION: 
        {student_question}
        """

        # 3. Prepare the inputs for Gemini
        contents = [prompt]
        if base64_image:
            clean_base64 = base64_image.split(",")[1]
            contents.append(
                types.Part.from_bytes(
                    data=base64.b64decode(clean_base64),
                    mime_type='image/jpeg'
                )
            )

        # 4. Ask Gemini to generate the answer
        response = ai_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=contents
        )
        
        return jsonify({"answer": response.text})

    except Exception as e:
        print(f"Backend Error: {e}")
        return jsonify({"answer": "មានបញ្ហាបច្ចេកទេសបន្តិចបន្តួច សូមព្យាយាមម្តងទៀត។"})

if __name__ == '__main__':
    # Cloud servers assign a dynamic PORT, so we must use os.environ
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)