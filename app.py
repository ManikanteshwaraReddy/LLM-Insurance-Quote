from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from langchain_community.llms import Ollama
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the previously generated embeddings and data
with open("my_faiss1.pkl", 'rb') as f:
    data, embeddings = pickle.load(f)

# Normalize the embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Create FAISS index
d = embeddings.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(d)

# Add embeddings to the index
index.add(embeddings)

# Save the FAISS index
# faiss.write_index(index, "my_faiss1.index")

# Serialize the index
pkl = faiss.serialize_index(index)

# Save the serialized index
with open('faiss1.index', 'wb') as f:
    f.write(pkl)

def retrieve(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return data.iloc[indices[0]]

def generate_response(combined_docs, query):
    # prompt_template = f"""
    
    # Act as a chat system for Health Insurance domain take as which input health insurance parameters and output Quotes for Insurance.
    # Based on the sample insurance data provide a quotation for health insurance based on input health condition of the user.

    #  medical_history,  charges:
    # sample health insurance data: {combined_docs}

    # ---
    # Act as a chat bot for Health Insurance domain.Be interactive with the user.
    # Ask the user to provide the neccessary health insurance parameters.Take the required parameters step by step.
    # Provide a Quotation in Rupees for the input health condition of the user based on the above sample health insurance data: 
    # input health condition of the user: {query}
    # """

    prompt_template = f"""
Act as a chat system for the Health Insurance domain. Take the necessary input health insurance parameters from the user and provide quotes for insurance. 

Below is some sample health insurance data that includes medical history and charges:
Sample health insurance data: {combined_docs}

---
Be interactive and guide the user step-by-step to provide the necessary health insurance parameters. Ask the user for each required parameter one at a time, and then provide a quotation in Rupees based on the input health condition of the user and the above sample health insurance data.

Once all the necessary information is collected, provide a health insurance quotation in Rupees based on the user's input and the sample health insurance data.

input health condition of the user: {query}
"""

    llm = Ollama(model="mistral:7b-instruct", temperature=0.3)
    response = llm.invoke(prompt_template)
        
    return response 

@app.route('/get-quote', methods=['POST'])
def get_quote():
    content = request.json
    age = content.get('age')
    gender = content.get('gender')
    bmi = content.get('bmi')
    children = content.get('children')
    smoker = content.get('smoker')
    medical_history = content.get('medical_history')
    family_medical_history = content.get('family_medical_history')
    exercise_frequency = content.get('exercise_frequency')

    #query = f"Age: {age}, Gender: {gender}, BMI: {bmi}, Children: {children}, Smoker: {smoker}, Medical History: {medical_history}, Family Medical History: {family_gemedical_history}, Exercise Frequency: {exercise_frequency}"
    query = f"generate a quote for health insurance based on the following health condition in detail: {medical_history}"
    retrieved_docs = retrieve(query)
    combined_docs = "\n".join(retrieved_docs)
    response = generate_response(combined_docs, query)
    
    return jsonify({"quotation": response})

if __name__ == '__main__':
    app.run(debug=False)
