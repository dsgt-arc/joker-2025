from config import bilingual, camembert, claude, deepseek, gemini, gemini_pro, gpt, mistral, o3, o4
from config import anthropic_key, deepseek_key, gemini_key, mistral_key, openai_key, openrouter_key
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from sentence_transformers import SentenceTransformer
from pydantic import SecretStr

import pandas as pd
import json


def get_response(prompt, model_str):
  # llm = ChatOpenRouter(model=get_api_model_str(model_str), api_key=openrouter_key)
  llm = get_model(model_str)
  response = llm.invoke(prompt).content

  print(response[response.find('{'):response.rfind('}') + 1].replace('\n', ''))
  response_json = json.loads(response[response.find('{'):response.rfind('}') + 1])
  return pd.Series(response_json)


def get_model(model_str):
  model = get_api_model_str(model_str)
  if model == o4 or model == o3 or model == gpt:
    return ChatOpenAI(model=model, api_key=openai_key)
  if model == gemini_pro or model == gemini:
    return ChatGoogleGenerativeAI(model=model, api_key=gemini_key)
  if model == claude:
    return ChatAnthropic(model=model, api_key=anthropic_key)
  if model == mistral:
    return ChatMistralAI(model=model, api_key=mistral_key)
  if model == deepseek:
    return ChatDeepSeek(model=model, api_key=deepseek_key)
  if model == camembert:
    return SentenceTransformer(camembert)
  if model == bilingual:
    return SentenceTransformer(bilingual, trust_remote_code=True)
  return ChatOpenRouter(model_name=model, openrouter_api_key=openrouter_key)


def get_api_model_str(model):
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
  if model == 'deepseek':
    return deepseek
  if model == 'camembert':
    return camembert
  if model == 'bilingual':
    return bilingual


class ChatOpenRouter(ChatOpenAI):
    openai_api_base: str
    openai_api_key: str
    model_name: str

    def __init__(self, model: str, api_key: str, openai_api_base: str = "https://openrouter.ai/api/v1"):
        super().__init__(openai_api_base=openai_api_base,
                         model_name=model)
