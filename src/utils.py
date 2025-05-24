from config import openai_key
from langchain_openai import ChatOpenAI
import pandas as pd
import json


def get_response(prompt, model):
    llm = ChatOpenAI(model=model, api_key=openai_key)
    response = llm.invoke(prompt).content

    print(response[response.find("{") : response.rfind("}") + 1].replace("\n", ""))
    response_json = json.loads(response[response.find("{") : response.rfind("}") + 1])
    return pd.Series(response_json)


def get_response_not_json(prompt, model):
    llm = ChatOpenAI(model=model, api_key=openai_key)
    response = llm.invoke(prompt).content
    return response
