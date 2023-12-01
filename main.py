import os,json,uuid,logging,httpx 
from time import time
from math import trunc
from typing import Annotated
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.security import HTTPBasic
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from Deepening_bot import DeepeningSalesGPT
from Project_info_bot import ProjectInfoSalesGPT
from functionalcalling import functions
from helpers import verify_username
from datetime import datetime

# Create a 'logs' directory if it doesn't already exist
if not os.path.exists("logs"):
    os.mkdir("logs")

# Load environment variables from a .env file 
load_dotenv()

# Define HTTP Basic Authentication security
security = HTTPBasic()

# Create a FastAPI application
app = FastAPI()

# Fetch the values from environment variables
correct_username = os.getenv("USER", "admin")
correct_password = os.getenv("PASSWORD", "1q2w3E*")

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4", temperature=0)

# List of allowed origins for cross-origin requests
origins = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

# Add CORS middleware to the FastAPI application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize all variables
session_dict_deepening = {}
session_dict_project_info = {}
total_tokens = 0
total_cost = 0
BOT_TOKEN = "fb772458-1394-4d56-a0c4-a37f17025a0b"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# All APIS

@app.get("/seeding_sales_conversation")
async def seeding_agent():
    # Generating uuid for new tab windows
    session_id = str(uuid.uuid4())
    session_log_file = f"logs/{timestamp}_{correct_username}_{session_id}.log"
    logging.basicConfig(
        filename=session_log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        encoding='utf8',
    )
    prev_stage = "1"
    # Initializing Bot 1
    deepening_sales_agent = DeepeningSalesGPT.from_llm(llm, verbose=False)
    deepening_sales_agent.seed_agent()

    # Initializing Bot 2
    project_info_sales_agent = ProjectInfoSalesGPT.from_llm(llm)
    project_info_sales_agent.seed_agent()

    session_dict_deepening[session_id] = [deepening_sales_agent, prev_stage]
    session_dict_project_info[session_id] = project_info_sales_agent
    logging.info("Session created: %s", session_id)
    return session_id

@app.post("/sales_conversation")
async def get_response(
    request: Request, query, username: str = Depends(verify_username)
):
    global total_tokens, total_cost, session_dict_deepening
    session_id = request.headers.get("Session-Id")
    sales_agent = session_dict_deepening[session_id][0]
    prev_stage = session_dict_deepening[session_id][1]

    if prev_stage == "6":
        return project_info_bot(query, session_id)

    sales_agent.human_step(query)
    
#function calling 
    isfunction = llm.predict_messages(
        [HumanMessage(content=query)], functions=functions
    )

    if isfunction.additional_kwargs:
        name = json.loads(
            isfunction.additional_kwargs["function_call"]["arguments"]
        ).get("name")
        number = json.loads(
            isfunction.additional_kwargs["function_call"]["arguments"]
        ).get("number")
        email = json.loads(
            isfunction.additional_kwargs["function_call"]["arguments"]
        ).get("email")
        logging.info(f"{name}, {email}, {number}")

    deepening_st = sales_agent.determine_conversation_stage()
    print(f"Bot1(Current conversation stage) = {deepening_st[0]} and previous conversation stage {prev_stage}")
    session_dict_deepening[session_id][1] = deepening_st[0]
    logging.info("Session ID: %s, Conversation Stage: %s", session_id, deepening_st)

    response = sales_agent.step()

    total_tokens += deepening_st[1] + response[1]
    total_cost += deepening_st[2] + response[2]
    logging.info(f"Total cost {total_cost}")
    logging.info(f"Total tokens {total_tokens}")
    logging.info(f"Total tokens {total_tokens}")
    logging.info("User %s queried: %s", username, query)
    logging.info("Response sent: %s", response[0])
    return response[0]


@app.post("/jivo/{id}")
async def get_from_jivochat(request: Request, id):
    if id != BOT_TOKEN:
        logging.error("Unauthorized request")
        raise HTTPException(status_code=401, detail="Not authenticated")
    data = await request.json()
    user_query = data["message"]["text"]
    bot_response = llm.predict(
        f"You are sales person so answer accordingly\n{user_query}"
    )
    internal_req = {
        "id": data["id"],
        "client_id": data["client_id"],
        "chat_id": data["chat_id"],
        "message": {
            "type": "TEXT",
            "text": bot_response,
            "timestamp": trunc(time.time()),
        },
        "event": "BOT_MESSAGE",
    }
    logging.info(f"Jivo Chat id - {data['id']}")
    logging.info(f"User query - {user_query}")
    logging.info(f"Our response - {bot_response}")
    res = await send_response(internal_req)
    logging.info(res)
    return res


# A function to send a response to the external service
async def send_response(response_data: dict):
    response_url = "https://bot.jivosite.com/webhooks/iFUyi7bwfxJgWZx/fb772458-1394-4d56-a0c4-a37f17025a0b"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(response_url, json=response_data)
            response.raise_for_status()  # Raise an exception if the request successful
            return response.text
        except httpx.HTTPError as e:
            logging.error(e)
            return f"Failed to send response: {str(e)}"

def project_info_bot(query, session_id, username: str = Depends(verify_username)):
    global total_tokens, total_cost, session_dict_project_info
    sales_agent = session_dict_project_info[session_id]

    project_info_st = sales_agent.determine_conversation_stage()
    print(f"Bot2 {project_info_st}")
    logging.info(f"Session ID: {session_id}, Conversation Stage: {project_info_st}")
    sales_agent.human_step(query)
    response = sales_agent.step()

    total_tokens += project_info_st[1] + response[1]
    total_cost += project_info_st[2] + response[2]
    logging.info(f"Total cost {total_cost}")
    logging.info(f"Total tokens {total_tokens}")
    logging.info("User %s queried: %s", username, query)
    logging.info("Response sent: %s", response[0])
    return response[0]

class AuthStaticFiles(StaticFiles):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    async def __call__(self, scope, receive, send) -> None:
        assert scope["type"] == "http"
        request = Request(scope, receive)
        await verify_username(request)
        await super().__call__(scope, receive, send)


app.mount("/", AuthStaticFiles(directory="frontend", html=True), name="frontend")
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
