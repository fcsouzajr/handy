from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel 
import re

app = FastAPI()

# Instância do modelo de degravação
model = WhisperModel("large-v3", device="cpu")


@app.post("/processar-audio/")
async def processar_audio(arquivo: UploadFile = File(...)):
    # Verifica se o arquivo enviado é de áudio
    if not arquivo.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="O arquivo enviado não é um arquivo de áudio válido.")

    # Carrega o arquivo de áudio em memória
    dados_audio = BytesIO(arquivo.file.read())

    try:
        segments, info = model.transcribe(dados_audio, language="pt")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na transcrição: {str(e)}")

    texts = [segment.text for segment in segments]
    resultado_transcricao = ''.join(texts).strip()

    # Retorna o resultado da transcrição e da análise
    return {
        "transcricao": resultado_transcricao,
    }
