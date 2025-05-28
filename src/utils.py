from config import claude, deepseek, gemini, gpt, mistral, o3
from config import openai_key, gemini_key, anthropic_key, deepseek_key, mistral_key
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

import pandas as pd
import json


def get_response(prompt, model):
  llm = get_llm(get_model(model))
  response = llm.invoke(prompt).content

  print(response[response.find('{'):response.rfind('}') + 1].replace('\n', ''))
  response_json = json.loads(response[response.find('{'):response.rfind('}') + 1])
  return pd.Series(response_json)


def get_model(model):
  if model == 'claude':
    return claude
  if model == 'deepseek':
    return deepseek
  if model == 'gemini':
    return gemini
  if model == 'gpt':
    return gpt
  if model == 'mistral':
    return mistral
  if model == 'o3':
    return o3


def get_llm(model):
  if model == claude:
    return ChatAnthropic(model=model, api_key=anthropic_key)
  if model == deepseek:
    return ChatDeepSeek(model=model, api_key=deepseek_key)
  if model == gemini:
    return ChatGoogleGenerativeAI(model=model, api_key=gemini_key)
  if model == gpt:
    return ChatOpenAI(model=model, api_key=openai_key)
  if model == mistral:
    return ChatMistralAI(model=model, api_key=mistral_key)
  if model == o3:
    return ChatOpenAI(model=model, api_key=openai_key)
