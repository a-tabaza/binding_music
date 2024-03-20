from vllm import LLM, SamplingParams
import json
import re

llm = LLM(model="./Mistral-7B-Instruct-v0.2-AWQ", quantization="awq", dtype="auto")

prompt_template = '''<s>[INST] {prompt} [/INST]'''

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)

instruction = '''
Extract metadata from the following song lyrics.
{lyrics}

Output it in JSON format, it must adhere to the following schema:
The key: "context", the value will be the a list of keywords that describes the paragraph, the list will have 5 keywords ONLY.
The key: "summary", the value will be the summary or a short synopsis of the lyrics, the main theme.
They key: "emotional_context", the value will be a list of keywords that describe the emotions within the song, the list will have 5 keywords ONLY.

Do NOT under any circumstances, output anything that can't be parsed into valid JSON.
Adhere to a word limit of 512 words MAX for any paragraph you generate.
You will be penalized if the JSON doesn't parse or if the word limit is exceded.
'''

from pydantic import BaseModel
import json

from fastapi import FastAPI, HTTPException, Body, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer

app = FastAPI(docs_url="/")

# middlewares
app.add_middleware(
    CORSMiddleware, # https://fastapi.tiangolo.com/tutorial/cors/
    allow_origins=['*'], # wildcard to allow all, more here - https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Origin
    allow_credentials=True, # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Credentials
    allow_methods=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Methods
    allow_headers=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Headers
)

api_keys = [
    "543c7086-c880-45de-8bce-6c9c906293bb"
]  

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def api_key_auth(api_key: str = Depends(oauth2_scheme)):
    if api_key not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden"
        )

class MistralInput(BaseModel):
  text: str


def clean_lyrics(lyrics):
    cleaned_lyrics = re.sub(r'\[.*?\]', '', lyrics)
    return cleaned_lyrics

@app.post('/metadata', dependencies=[Depends(api_key_auth)])
async def get_metadata(user_input: MistralInput):
  lyrics = user_input.text
  cleaned_lyrics = clean_lyrics(lyrics)
  try:
    outputs = llm.generate(prompt_template.format(prompt=instruction.format(lyrics = cleaned_lyrics)), sampling_params)
    return json.loads(outputs[0].outputs[0].text)
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"{e}")

