from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from courserec_logic import recommend_courses  

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/recommend")
def recommend(prompt: str = Form(...)):
    results = recommend_courses(prompt)
    return {"courses": results}
