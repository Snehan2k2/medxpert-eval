from openai import OpenAI
from typing import List
import base64
import requests
from together import Together
import json

class LLMAgent:
    def __init__(self, temperature, max_tokens, num_gen) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_gen = num_gen
        pass
    def get_response(self):
        pass

def encode_image(image_path):
    if image_path:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    else:
        return "No image inputs"

class APIAgent(LLMAgent):
    def __init__(self, temperature, max_tokens, num_gen, port) -> None:
        super().__init__(temperature, max_tokens, num_gen)
        print("local hosting!")
        self.model_url = f"http://localhost:{port}/v1/chat/completions"

    def get_response(self, messages: List[dict]) -> str:
        if self.num_gen == 1:
            payload = {
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "n" : self.num_gen,
                "seed": 0,
            }
        else:
            payload = {
                "messages": messages,
                "max_tokens": self.max_tokens,
                "n" : self.num_gen,
                "seed": 0,
            }
        headers = {
            "Content-Type": "application/json"
        }
        res = requests.post(self.model_url, json=payload, headers=headers)
        res.raise_for_status()
        result = res.json()
        response = []
        for i in range(self.num_gen):
            response.append(result["choices"][i]["message"]["content"])
        return response

    def image_content(self, img_path: str) -> dict:
        img_path = img_path.strip()
        if img_path.startswith("http"):
            return {"type": "image_url", "image_url": {"url": img_path}}
        else:
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}}
