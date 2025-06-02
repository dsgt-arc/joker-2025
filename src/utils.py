from config import camembert, claude, gemini, gemini_pro, gpt, mistral, o3, o4, google
from config import openai_key, gemini_key, anthropic_key, mistral_key
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

import pandas as pd
import json

def get_response(prompt, model):
  llm = get_llm(get_model(model))
  response = llm.invoke(prompt).content

  print(response[response.find('{'):response.rfind('}') + 1].replace('\n', ''))
  response_json = json.loads(response[response.find('{'):response.rfind('}') + 1])
  return pd.Series(response_json)

def get_response_not_json(prompt, model):
   llm = get_llm(get_model(model))
   response = llm.invoke(prompt).content
   return response

def get_model(model):
  if model == 'o4':
    return o4
  if model == 'o3':
    return o3
  if model == 'gpt':
    return gpt
  if model == 'gemini_pro':
    return gemini_pro
  if model == 'gemini':
    return gemini
  if model == 'claude':
    return claude
  if model == 'mistral':
    return mistral
  if model == 'camembert':
    return camembert


def get_llm(model):
  if model == o4 or model == o3 or model == gpt:
    return ChatOpenAI(model=model, api_key=openai_key)
  if model == gemini_pro or model == gemini:
    return ChatGoogleGenerativeAI(model=model, api_key=gemini_key)
  if model == claude:
    return ChatAnthropic(model=model, api_key=anthropic_key)
  if model == mistral:
    return ChatMistralAI(model=model, api_key=mistral_key)
  if model == camembert:
    return SentenceTransformer(camembert)
