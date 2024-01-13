import sys
import os
import time
import uvicorn
from model import OpenAIModel
from translator import PDFTranslator
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

sys_path = os.path.dirname(os.path.abspath(__file__))  # 得到 ai_translator/ 目录
sys.path.append(sys_path)

app = FastAPI()


# 静态文件中间件，用于处理静态文件，如HTML、CSS、JS等
root_dir = os.path.dirname(sys_path)   # 得到 openai-translator/ 目录
print(f"root_dir={root_dir}")
app.mount("/static", StaticFiles(directory=os.path.join(root_dir, "statics")), name="statics")


def ensure_dir(file_dir: str):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


# 处理上传文件的路由
@app.post("/translate")
async def translate_file(file: UploadFile = File(...), target_language: str = Form(...), file_format: str = Form(...)):
    # 获取上传的文件名（不包括路径）
    origin_filename = file.filename
    print(f"request params --> origin_filename: {origin_filename}, target_language: {target_language}")

    # 保存文件到服务器的本地文件系统中，这里我们将其保存在临时目录中
    ensure_dir(os.path.join(root_dir, "temp"))
    temp_file_name = f"{str(int(time.time()))}____{origin_filename}"
    temp_file = os.path.join(root_dir, "temp", temp_file_name)
    with open(temp_file, "wb") as f:
        f.write(await file.read())

    model_name = os.getenv("model_name")
    api_key = os.getenv("openai_api_key")
    api_base_url = os.getenv("api_base_url")
    model = OpenAIModel(model=model_name, api_key=api_key, base_url=api_base_url)

    # 实例化 PDFTranslator 类，并调用 translate_pdf() 方法
    translator = PDFTranslator(model)
    translated_file = translator.translate_pdf(temp_file, file_format, target_language=target_language)

    # return {"filename": origin_filename}
    return StreamingResponse(open(translated_file, "rb"))

# 主路由，返回一个简单的HTML页面，用于上传文件
@app.get("/")
async def read_root():
    return FileResponse(os.path.join(root_dir, "statics", "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
