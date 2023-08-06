from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import TextLoader
import os
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
from fastapi import FastAPI, HTTPException
import firebase_admin
from firebase_admin import credentials, auth
import pyrebase
from pyrebase import initialize_app
import json
from dotenv import load_dotenv
import random
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain

load_dotenv(".env")

openai_key = os.getenv('OPENAI_API_KEY')

firebaseConfig = {
    "apiKey": "AIzaSyDrlc4ScLE96wPXYPX7Zka_eSRsY32l7TQ",
    "authDomain": "apollov1-6ca8c.firebaseapp.com",
    "projectId": "apollov1-6ca8c",
    "storageBucket": "apollov1-6ca8c.appspot.com",
    "messagingSenderId": "79939313885",
    "appId": "1:79939313885:web:d88b1370b840f8b15dfb34",
   "measurementId": "G-ZYJ4HZC83P",
   "databaseURL": ""
  }


# setting up firebase
cred = credentials.Certificate("apollov1-6ca8c-firebase-adminsdk-6ejxf-449afd9ae4.json")
firebase_admin.initialize_app(cred)
pb = pyrebase.initialize_app(firebaseConfig)
db = pb.database()

app=FastAPI()

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"]
                   )

class QuestionAnswer(BaseModel):
    question: str

class PromptModel(BaseModel):
    query: str

class PubModel(BaseModel):
    input_specification: str 
    operation_specification: str

class CreateUserRequest(BaseModel):
    email: str
    password: str
    firstName: str
    lastName: str
    licenseLevel: str
    country: str
    state: str
    dpname: str
    photo: str
    phone: str
    localProtocol: str
    subscriptionInfo: str


class UpdateUserRequest(BaseModel):
    email: str = None
    password: str = None
    userProfile: dict = None

class ThumbsCount(BaseModel):
    id: str


root_dir = "./data"

# Print number of txt files in directory
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith(".docx") or file.endswith('.csv'):
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
# initialize pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
)

index_name = "medicbot"

# First, check if our index already exists. If it doesn't, we create it
if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
      name=index_name,
      metric='cosine',
      dimension=1536  
)
# The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)

chain = load_qa_chain(llm, chain_type="stuff")

def get_similiar_docs(query, k=2, score=False):
    if score:
        similar_docs = docsearch.similarity_search_with_score(query, k=k)
    else:
        similar_docs = docsearch.similarity_search(query, k=k)
    return similar_docs

@app.post("/similarity_search")
async def read_from_docs(query: PromptModel):
    similar_docs = get_similiar_docs(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer
    

@app.post("/answer_query")
async def fallback_chat_completion(query: PromptModel):
    openai.api_key = openai_key
    chat_history = [
        {
            "role": "system",
            "content": "Iam a proffesional health expert or a medic. I help with anyting regarding best practices of pills, precautions etc"
        },
        {
            "role": "user",
            "content": f"{query}"
        }
    ]
    # Call the GPT-3.5-turbo completion API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the GPT-3.5-turbo engine
        messages= chat_history ,
        temperature=0,
        max_tokens=150 
    )

    return response["choices"][0]["message"]


@app.get('/pubchem')
def get_pubmed_data(pubrequest: PubModel):

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/{pubrequest.input_specification}/{pubrequest.operation_specification}/JSON"
    response = requests.get(url)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Unable to fetch data from NLM API.")
    
    return response



# User management
@app.post('/create_user')
def create_user(request: CreateUserRequest):
    try:
        user = auth.create_user(
        email=request.email,
        email_verified=request.email,
        phone_number=request.phone,
        password= request.password,
        display_name=request.dpname,
        photo_url= request.photo,
        )
        return user
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while creating user: {e}")


@app.post('/login')
def login(request: dict):
    try:
        email = request.get('email')
        password = request.get('password')

        user = auth.sign_in_with_email_and_password(email, password)

        # Get the user ID and generate a JWT token if needed
        user_id = user['localId']
        token = auth.create_custom_token(user_id)

        return {'userId': user_id, 'token': token}, 200
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while logging in: {e}")


@app.delete('/user/{id}')
def delete_user(id: str):
    try:
        auth.delete_user(id)
        db.child('users').child(id).remove()
        return '', 204
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while deleting user: {e}")


@app.post('/sessionLogin')
def session_login(request: dict):
    try:
        id_token = request.get('idToken')
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token.get('uid')
        custom_claims = decoded_token.get('claims', {})
        return {'uid': uid, 'customClaims': custom_claims}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while logging in: {e}")


@app.get('/users')
def list_users():
    try:
        users = auth.list_users().iterate_all()
        users_list = [user.uid for user in users]
        return users_list
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred while listing users: {e}")


def recognize_pill(params_text):
    # This is just a placeholder function. In a real implementation, you would call the language model API.
    # For this example, we'll return some dummy values.
    recognized_attributes = {
        "type": random.choice(["capsule", "tablet"]),
        "shape": random.choice(["oval", "round", "square"]),
        "color": random.choice(["red", "blue", "white"]),
        "lettering": "Pill A",
        "numbering": "123",
    }
    return recognized_attributes

@app.post("/pill_recognition")
def pill_recognition(params: dict):
    try:
        # Preprocess the user's input parameters into a formatted text string.
        params_text = preprocess_params(params)

        # Call the language model to recognize pill attributes based on the input text.
        recognized_attributes = recognize_pill(params_text)

        # Validate the attributes and apply confidence thresholding if needed.
        validated_attributes = validate_attributes(recognized_attributes)

        return {"result": validated_attributes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during pill recognition: {e}")

def preprocess_params(params: dict) -> str:
    # Implement preprocessing logic here to convert the input parameters into a formatted text string.

    # Ensure all attribute keys are lowercase for consistency (optional step)
    params_lower = {key.lower(): value for key, value in params.items()}

    # Create the formatted text string
    params_text = "Pill recognition parameters:\n"
    params_text += f"Type: {params_lower.get('type', 'N/A')}\n"
    params_text += f"Shape: {params_lower.get('shape', 'N/A')}\n"
    params_text += f"Color: {params_lower.get('color', 'N/A')}\n"
    params_text += f"Lettering: {params_lower.get('lettering', 'N/A')}\n"
    params_text += f"Numbering: {params_lower.get('numbering', 'N/A')}\n"

    # You can add more attributes or modify the formatting as needed.

    return params_text


def validate_attributes(attributes: dict) -> dict:
    # Implement validation and confidence thresholding logic here for the recognized attributes.

    # Check if all required attributes are present
    required_attributes = ["type", "shape", "color", "lettering", "numbering"]
    for attribute in required_attributes:
        if attribute not in attributes:
            raise ValueError(f"Missing required attribute: {attribute}")

    # Perform additional checks and filtering based on confidence thresholds or specific rules
    # For example, you might want to ensure that the color value is a valid color name or code.
    # Here, we are using some placeholder checks.

    # Validate the color attribute
    valid_colors = ["red", "blue", "green", "white", "yellow", "black"]
    if "color" in attributes and attributes["color"].lower() not in valid_colors:
        raise ValueError("Invalid color value")

    # Validate the pill type attribute
    valid_pill_types = ["capsule", "tablet", "pill"]
    if "type" in attributes and attributes["type"].lower() not in valid_pill_types:
        raise ValueError("Invalid pill type")

    # Apply a confidence threshold for lettering and numbering attributes
    confidence_threshold = 0.7
    if "lettering" in attributes and attributes["lettering_confidence"] < confidence_threshold:
        attributes["lettering"] = None

    if "numbering" in attributes and attributes["numbering_confidence"] < confidence_threshold:
        attributes["numbering"] = None

    # If any validation or filtering is required for other attributes, add it here.

    # Return the validated attributes
    return attributes



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")

