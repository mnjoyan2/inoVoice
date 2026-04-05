import logging
import os
import tempfile

import numpy as np
import sherpa_onnx
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from omnivoice.cli.infer import get_best_device
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STT_MODEL_DIR = os.environ.get(
    "OMNILINGUAL_ASR_DIR",
    "/workspace/OmniVoice/asr_models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12",
)
TTS_MODEL_ID = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
STT_NUM_THREADS = int(os.environ.get("OMNILINGUAL_ASR_THREADS", "4"))
STT_PROVIDER = os.environ.get("OMNILINGUAL_ASR_PROVIDER", "cpu")

VOICES_DIR = os.environ.get("VOICES_DIR", "/workspace/voices")

VOICES = {
    "angry_marcus": {"emotion": "angry", "gender": "male", "pitch": "low", "age": "young adult"},
    "angry_diana": {"emotion": "angry", "gender": "female", "pitch": "moderate", "age": "young adult"},
    "angry_victor": {"emotion": "angry", "gender": "male", "pitch": "moderate", "age": "middle-aged"},
    "sad_elena": {"emotion": "sad", "gender": "female", "pitch": "low", "age": "young adult"},
    "sad_james": {"emotion": "sad", "gender": "male", "pitch": "low", "age": "middle-aged"},
    "sad_margaret": {"emotion": "sad", "gender": "female", "pitch": "low", "age": "elderly"},
    "happy_aria": {"emotion": "happy", "gender": "female", "pitch": "high", "age": "young adult"},
    "happy_leo": {"emotion": "happy", "gender": "male", "pitch": "high", "age": "young adult"},
    "happy_sophie": {"emotion": "happy", "gender": "female", "pitch": "moderate", "age": "teenager"},
    "calm_oliver": {"emotion": "calm", "gender": "male", "pitch": "moderate", "age": "middle-aged"},
    "calm_nina": {"emotion": "calm", "gender": "female", "pitch": "low", "age": "young adult"},
    "calm_henry": {"emotion": "calm", "gender": "male", "pitch": "low", "age": "elderly"},
    "excited_luna": {"emotion": "excited", "gender": "female", "pitch": "very high", "age": "young adult"},
    "excited_ryan": {"emotion": "excited", "gender": "male", "pitch": "high", "age": "young adult"},
    "excited_mia": {"emotion": "excited", "gender": "female", "pitch": "high", "age": "teenager"},
    "serious_grant": {"emotion": "serious", "gender": "male", "pitch": "low", "age": "middle-aged"},
    "serious_claire": {"emotion": "serious", "gender": "female", "pitch": "moderate", "age": "middle-aged"},
    "serious_walter": {"emotion": "serious", "gender": "male", "pitch": "very low", "age": "elderly"},
    "gentle_emma": {"emotion": "gentle", "gender": "female", "pitch": "low", "age": "young adult"},
    "gentle_thomas": {"emotion": "gentle", "gender": "male", "pitch": "moderate", "age": "middle-aged"},
    "gentle_rose": {"emotion": "gentle", "gender": "female", "pitch": "moderate", "age": "elderly"},
    "confident_alex": {"emotion": "confident", "gender": "male", "pitch": "moderate", "age": "young adult"},
    "confident_sarah": {"emotion": "confident", "gender": "female", "pitch": "high", "age": "young adult"},
    "confident_daniel": {"emotion": "confident", "gender": "male", "pitch": "low", "age": "middle-aged"},
    "tired_nathan": {"emotion": "tired", "gender": "male", "pitch": "low", "age": "young adult"},
    "tired_lisa": {"emotion": "tired", "gender": "female", "pitch": "low", "age": "young adult"},
    "tired_george": {"emotion": "tired", "gender": "male", "pitch": "low", "age": "elderly"},
    "neutral_sam": {"emotion": "neutral", "gender": "male", "pitch": "moderate", "age": "young adult"},
    "neutral_kate": {"emotion": "neutral", "gender": "female", "pitch": "moderate", "age": "young adult"},
    "neutral_david": {"emotion": "neutral", "gender": "male", "pitch": "moderate", "age": "middle-aged"},
}

_recognizer = None
_tts_model = None


def load_audio(path):
    samples, sample_rate = sf.read(path, dtype="float32")
    if samples.ndim > 1:
        samples = samples[:, 0]
    return samples, sample_rate


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        tokens_path = os.path.join(STT_MODEL_DIR, "tokens.txt")
        model_path = os.path.join(STT_MODEL_DIR, "model.int8.onnx")
        _recognizer = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=model_path,
            tokens=tokens_path,
            num_threads=STT_NUM_THREADS,
            decoding_method="greedy_search",
            debug=False,
            provider=STT_PROVIDER,
        )
        logger.info("STT recognizer ready")
    return _recognizer


def get_tts():
    global _tts_model
    if _tts_model is None:
        device = os.environ.get("OMNIVOICE_DEVICE") or get_best_device()
        load_asr = os.environ.get("OMNIVOICE_LOAD_ASR", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        logger.info("Loading TTS model %s on %s", TTS_MODEL_ID, device)
        _tts_model = OmniVoice.from_pretrained(
            TTS_MODEL_ID,
            device_map=device,
            dtype=torch.float16,
            load_asr=load_asr,
        )
        logger.info("TTS model ready")
    return _tts_model


def _resolve_language(language: str | None) -> str | None:
    if language is None:
        return None
    s = language.strip()
    if not s or s.lower() == "auto":
        return None
    return s


def _cleanup_task(*paths):
    def _clean():
        for p in paths:
            if p and os.path.exists(p):
                os.unlink(p)
    return BackgroundTask(_clean)


app = FastAPI(title="Speech API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "stt_model_dir": STT_MODEL_DIR, "tts_model": TTS_MODEL_ID}


@app.get("/api/voices")
def list_voices():
    return {
        "voices": [
            {"name": name, **meta}
            for name, meta in VOICES.items()
        ]
    }


@app.get("/api/voices/{name}/sample")
def voice_sample(name: str):
    if name not in VOICES:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    path = os.path.join(VOICES_DIR, f"{name}.wav")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Sample file for '{name}' not found")
    return FileResponse(path, media_type="audio/wav", filename=f"{name}.wav")


@app.post("/api/stt")
async def transcribe(audio: UploadFile = File(...)):
    data = await audio.read()
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        path = f.name
    try:
        recognizer = get_recognizer()
        samples, sample_rate = load_audio(path)
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        return {"text": text}
    finally:
        os.unlink(path)


@app.post("/api/tts")
async def synthesize(
    text: str = Form(...),
    voice: str | None = Form(None),
    language: str | None = Form(None),
    instruct: str | None = Form(None),
    ref_text: str | None = Form(None),
    ref_audio: UploadFile | None = File(None),
    num_step: int = Form(32),
    guidance_scale: float = Form(2.0),
    t_shift: float = Form(0.1),
    denoise: bool = Form(True),
    speed: float = Form(1.0),
    duration: float | None = Form(None),
    preprocess_prompt: bool = Form(True),
    postprocess_output: bool = Form(True),
    layer_penalty_factor: float = Form(5.0),
    position_temperature: float = Form(5.0),
    class_temperature: float = Form(0.0),
    audio_chunk_duration: float = Form(15.0),
    audio_chunk_threshold: float = Form(30.0),
):
    model = get_tts()
    ref_path = None
    try:
        if ref_audio is not None:
            rsfx = os.path.splitext(ref_audio.filename or "")[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=rsfx) as f:
                f.write(await ref_audio.read())
                ref_path = f.name
        elif voice and voice.strip():
            voice_name = voice.strip()
            if voice_name not in VOICES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown voice '{voice_name}'. Use GET /api/voices to list available voices.",
                )
            voice_path = os.path.join(VOICES_DIR, f"{voice_name}.wav")
            if not os.path.isfile(voice_path):
                raise HTTPException(status_code=500, detail=f"Voice file for '{voice_name}' is missing on disk")
            ref_path = voice_path

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale)
            if guidance_scale is not None
            else 2.0,
            t_shift=float(t_shift),
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
            layer_penalty_factor=float(layer_penalty_factor),
            position_temperature=float(position_temperature),
            class_temperature=float(class_temperature),
            audio_chunk_duration=float(audio_chunk_duration),
            audio_chunk_threshold=float(audio_chunk_threshold),
        )

        kw: dict = {
            "text": text.strip(),
            "language": _resolve_language(language),
            "generation_config": gen_config,
        }

        if ref_path is not None:
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_path,
                ref_text=ref_text or None,
                preprocess_prompt=gen_config.preprocess_prompt,
            )
        elif instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        logger.info(
            "TTS request: text_len=%d, language=%s, voice=%s, ref=%s",
            len(text),
            kw.get("language"),
            voice,
            ref_path is not None,
        )

        audios = model.generate(**kw)

        waveform = audios[0].squeeze(0).cpu().numpy()
        waveform_int16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, waveform_int16, model.sampling_rate, format="WAV", subtype="PCM_16")

        logger.info(
            "TTS done: duration=%.1fs",
            len(waveform_int16) / model.sampling_rate,
        )

        is_voice_lib = ref_path and ref_path.startswith(VOICES_DIR)
        cleanup_paths = [tmp.name] if is_voice_lib else [tmp.name, ref_path]

        return FileResponse(
            tmp.name,
            media_type="audio/wav",
            filename="output.wav",
            background=_cleanup_task(*cleanup_paths),
        )
    except HTTPException:
        raise
    except Exception:
        if ref_path and not ref_path.startswith(VOICES_DIR) and os.path.exists(ref_path):
            os.unlink(ref_path)
        raise


@app.on_event("startup")
def startup():
    if os.environ.get("PRELOAD_STT", "").lower() in ("1", "true", "yes"):
        get_recognizer()
    if os.environ.get("PRELOAD_TTS", "").lower() in ("1", "true", "yes"):
        get_tts()
