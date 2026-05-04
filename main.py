from llama_cpp import Llama
from pydantic import BaseModel
from fastapi import FastAPI
llm = Llama(model_path="./model.gguf", n_gpu_layers=-1)
# n_gpu_layers=-1 ile bütün yükü GPU'ya yıktım

app = FastAPI(title="Dream-Catcher-Server", version="1.0")

class DreamRequest(BaseModel):
    dream_text: str

@app.get("/status")
def server_status():
    return {"status": "Dream server is working"}

@app.post("/analyze")
def analyze_dream(dream_request: DreamRequest):
    text = dream_request.dream_text

    format = f"""### Instruction: You are a proffesional dream analyzer and know Freud and Jung opinions very well. Analyze the concept below and give me the meaning of the dream, 
    ### Input: {text}
    ### Output:
    """ 

    output = llm(format, 
                 max_tokens=256,
                 stop=["<|eot_id|>"],
                 temperature=0.3,
                 top_p=0.9,
                 repeat_penalty=1.2)
    
    return {"dream ": text, 
            "answer": output['choices'][0]["text"]}
    