from flask import Flask, request
from generate import agent_executor
from dotenv import load_dotenv 
from urllib.parse import urlparse
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor
from langchain import OpenAI
from flask_cors import CORS, cross_origin
import requests
import os
import uuid

load_dotenv()

app = Flask(__name__)
CORS(app,)


@app.route('/')
def home():
    return "Flask up and running!"


# for legal ai upload pdf 
@cross_origin('*')
@app.route('/legal-ai-upload')
def legal_ai_upload():

    if request.method == 'POST':
        url = request.json['pdfurl']
        parsedurl = urlparse(url)

        pdf = os.path.basename(parsedurl.path)
        pdf_ID = str(uuid.uuid4())   # unique identifier for each user
        response = requests.get(url)

        if not os.path.exists(f'static/pdfs/{pdf}'):
            os.makedirs(f'static/pdfs/{pdf_ID}')

        # Check if the request was successful
        if response.status_code == 200:
            with open(f'static/pdfs/{pdf_ID}/{pdf}.pdf', 'wb') as f:
                f.write(response.content)

        return {"pdf_ID":pdf_ID}


# to chat with the uploaded legal pdf
@cross_origin(supports_credentials=True)
@app.route('/legal-ai-chat')
def legal_ai_chat():
    f_path = request.json["f_path"] if request.json["f_path"] else ""
    query = request.json["message"] if request.json["message"] else ""

    if os.path.exists(f'static/index/{f_path}.json'):
        print("Loading Index loop")

        # load from disk
        loaded_index = GPTSimpleVectorIndex.load_from_disk(f'static/index/{f_path}.json')
        response = loaded_index.query(query, verbose=True, response_mode="default")
        final_answer = str(response)
        return {"answer":final_answer}
    
    else:
        print("Creating Index loop")
        
        # Set path of indexed jsons
        index_path = f"static/index/{f_path}.json"

        documents = SimpleDirectoryReader(f'static/pdfs/{f_path}').load_data()

        # builds an index over the documents in the data folder
        index = GPTSimpleVectorIndex(documents)

        # save the index to disk
        index.save_to_disk(index_path)

        # define the llm to be used
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

        # load from disk
        loaded_index = GPTSimpleVectorIndex.load_from_disk(index_path, llm_predictor=llm_predictor)
        response = loaded_index.query(query, verbose=True, response_mode="default")

        final_answer = str(response)

        return {"answer":final_answer}
    



















@app.route('/chat')
def chat():

    query = request.json['query'] if request.json['query'] else ''
    # put the logic here for intermediate code
    final_answer = agent_executor.run(query)
    return {"response": final_answer}


if __name__ == '__main__':
    app.run(debug=True)