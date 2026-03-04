#!/usr/bin/env python3
"""
セカンドブレーン - ローカルサーバー（WhisperX + Moonshine JA + Azure OpenAI）

依存:
  pip install whisperx transformers torch openai
  brew install ffmpeg  (macOS) / apt install ffmpeg  (WSL)

話者分離を使う場合は HuggingFace Token も必要

使い方:
  python server.py
  → ブラウザで http://localhost:8080 を開く
  → 画面右上の ⚙ からAPIキー・HFトークン・Azure設定を入力
"""

import http.server
import json
import urllib.request
import urllib.error
import os
import sys
import tempfile
import threading

PORT = 8080

# ──────────────────────────────────────────
# デバイス検出
# ──────────────────────────────────────────
# ffmpeg の場所を PATH に追加（venv 内 / Homebrew / システム）
_here = os.path.dirname(os.path.abspath(__file__))
_venv_bin = os.path.join(_here, "venv", "bin")
os.environ["PATH"] = _venv_bin + ":/opt/homebrew/bin:" + os.environ.get("PATH", "")

GPU_NAME = ""
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
        COMPUTE_TYPE = "float16"
        WHISPER_MODEL = "large-v3"   # WSL GPU 環境
        GPU_NAME = torch.cuda.get_device_name(0)
        print(f"  [Device] CUDA GPU: {GPU_NAME}", flush=True)
    else:
        DEVICE = "cpu"
        COMPUTE_TYPE = "int8"
        WHISPER_MODEL = "base"       # CPU 環境（Mac 等）
        print("  [Device] CPU (int8)", flush=True)
except Exception:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"
    WHISPER_MODEL = "base"

# ──────────────────────────────────────────
# WhisperX モデル管理
# ──────────────────────────────────────────
_whisper_model  = None
_diarize_model  = None
_model_lock     = threading.Lock()

ALLOWED_MODELS = {"pyannote/speaker-diarization-3.1"}
DEFAULT_MODEL  = "pyannote/speaker-diarization-3.1"


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisperx
        except ImportError:
            raise RuntimeError(
                "whisperx がインストールされていません。\n"
                "  pip install whisperx\nを実行してください。"
            )
        print(f"  [WhisperX] モデル読み込み中 ({WHISPER_MODEL}, ja)...", flush=True)
        _whisper_model = whisperx.load_model(
            WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE, language="ja",
            vad_options={"vad_onset": 0.2, "vad_offset": 0.1},
        )
        print("  [WhisperX] 完了", flush=True)
    return _whisper_model


def _get_diarize(hf_token, model_name=None):
    global _diarize_model
    if not hf_token:
        return None
    target = model_name if model_name in ALLOWED_MODELS else DEFAULT_MODEL
    if _diarize_model is None:
        from whisperx.diarize import DiarizationPipeline
        print(f"  [WhisperX] 話者分離モデル読み込み中: {target}", flush=True)
        _diarize_model = DiarizationPipeline(
            model_name=target, token=hf_token, device=DEVICE
        )
        print("  [WhisperX] 話者分離完了", flush=True)
    return _diarize_model


def transcribe_audio(audio_path, hf_token="", diarize_model_name=None, initial_prompt=""):
    import whisperx
    with _model_lock:
        whisper_model = _get_whisper()
        diarize_model = _get_diarize(hf_token, diarize_model_name)

    audio  = whisperx.load_audio(audio_path)
    result = whisper_model.transcribe(
        audio, batch_size=4,
        initial_prompt=initial_prompt or None
    )
    if not result.get("segments"):
        return []

    if diarize_model is not None:
        diarize_segs = diarize_model(audio)
        result       = whisperx.assign_word_speakers(diarize_segs, result)
        return [
            {"speaker": seg.get("speaker", "SPEAKER_00"),
             "text":    seg.get("text", "").strip()}
            for seg in result.get("segments", [])
            if seg.get("text", "").strip()
        ]
    else:
        full_text = " ".join(
            s.get("text", "").strip() for s in result.get("segments", [])
        ).strip()
        return [{"speaker": "SPEAKER_00", "text": full_text}] if full_text else []


# ──────────────────────────────────────────
# Moonshine JA モデル管理（pipeline不使用・torchcodec回避）
# ──────────────────────────────────────────
_moonshine_model     = None
_moonshine_processor = None
_moonshine_lock      = threading.Lock()
MOONSHINE_MODEL      = "UsefulSensors/moonshine-tiny-ja"

# venv 内の ffmpeg バイナリを優先（torchcodec に依存しない）
_FFMPEG = os.path.join(_venv_bin, "ffmpeg")
if not os.path.exists(_FFMPEG):
    _FFMPEG = "ffmpeg"


def _get_moonshine():
    global _moonshine_model, _moonshine_processor
    if _moonshine_model is None:
        try:
            import torch
            from transformers import MoonshineForConditionalGeneration, AutoProcessor
        except ImportError:
            raise RuntimeError(
                "transformers がインストールされていません。\n"
                "  pip install transformers torch\nを実行してください。"
            )
        print(f"  [Moonshine] モデル読み込み中: {MOONSHINE_MODEL} ...", flush=True)
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        _moonshine_processor = AutoProcessor.from_pretrained(MOONSHINE_MODEL)
        _moonshine_model = MoonshineForConditionalGeneration.from_pretrained(
            MOONSHINE_MODEL, torch_dtype=dtype
        )
        if DEVICE == "cuda":
            _moonshine_model = _moonshine_model.cuda()
        _moonshine_model.eval()
        print("  [Moonshine] 完了", flush=True)
    return _moonshine_model, _moonshine_processor


def _load_audio_ffmpeg(audio_path):
    """ffmpeg で 16kHz mono float32 numpy 配列に変換（torchcodec を使わない）"""
    import subprocess
    import numpy as np

    cmd = [_FFMPEG, "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg 変換失敗: " + proc.stderr.decode(errors="replace")[:300])
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def transcribe_moonshine_ja(audio_path):
    """Moonshine JA で文字起こし → [{"speaker": "SPEAKER_00", "text": "..."}]"""
    import torch

    # 音声を numpy 配列に変換（torchcodec を経由しない）
    try:
        import whisperx
        audio = whisperx.load_audio(audio_path)   # 16kHz mono float32
    except Exception:
        audio = _load_audio_ffmpeg(audio_path)

    with _moonshine_lock:
        model, processor = _get_moonshine()

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    if DEVICE == "cuda":
        inputs = {k: v.to("cuda", dtype=torch.float16 if v.is_floating_point() else v.dtype)
                  for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,        # 30秒で日本語200文字が上限の目安
            repetition_penalty=1.3,    # 繰り返しループを抑制
            no_repeat_ngram_size=5,    # 5-gram の繰り返しを禁止
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return [{"speaker": "SPEAKER_00", "text": text}] if text else []


# ──────────────────────────────────────────
# HTTP ハンドラ
# ──────────────────────────────────────────
class SecondBrainHandler(http.server.SimpleHTTPRequestHandler):
    """静的ファイル配信 + WhisperX + Moonshine JA + Anthropic / Azure OpenAI プロキシ"""

    # ─── GET ──────────────────────────────
    def do_GET(self):
        if self.path == "/api/info":
            import time as _time
            base   = os.path.dirname(os.path.abspath(__file__))
            mtimes = [os.path.getmtime(os.path.join(base, f))
                      for f in ("server.py", "index.html")
                      if os.path.exists(os.path.join(base, f))]
            mtime  = max(mtimes) if mtimes else 0
            self._json_response(200, {
                "device":         DEVICE,
                "compute_type":   COMPUTE_TYPE,
                "whisper_model":  WHISPER_MODEL,
                "gpu_name":       GPU_NAME,
                "moonshine_model": MOONSHINE_MODEL,
                "updated_at":     _time.strftime("%m/%d %H:%M", _time.localtime(mtime)),
            })
        else:
            super().do_GET()

    # ─── POST ─────────────────────────────
    def do_POST(self):
        if   self.path == "/api/transcribe":
            self._handle_transcribe()
        elif self.path == "/api/moonshine-transcribe":
            self._handle_moonshine_transcribe()
        elif self.path == "/api/analyze":
            self._proxy_openai()
        else:
            self.send_error(404)

    # ─── /api/transcribe (WhisperX) ───────
    def _handle_transcribe(self):
        try:
            length   = int(self.headers.get("Content-Length", 0))
            hf_token = self.headers.get("X-HF-Token", "").strip()
            dmodel   = self.headers.get("X-Diarize-Model", "").strip() or None
            vocab    = self.headers.get("X-Vocab", "").strip()
            audio_bytes = self.rfile.read(length)

            ctype  = self.headers.get("Content-Type", "audio/webm").split(";")[0].strip()
            ext_map = {"audio/webm":".webm","audio/ogg":".ogg","audio/mp4":".mp4","audio/mpeg":".mp3"}
            suffix = ext_map.get(ctype, ".webm")

            initial_prompt = ""
            if vocab:
                terms = "、".join(t.strip() for t in vocab.splitlines() if t.strip())
                initial_prompt = f"本会話では次の専門用語が登場します：{terms}。"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            try:
                segments = transcribe_audio(tmp_path, hf_token, dmodel, initial_prompt)
                self._json_response(200, {"segments": segments})
            finally:
                try: os.unlink(tmp_path)
                except OSError: pass

        except Exception as e:
            self._json_response(500, {"error": f"文字起こしエラー: {e}"})

    # ─── /api/moonshine-transcribe ────────
    def _handle_moonshine_transcribe(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            audio_bytes = self.rfile.read(length)

            ctype  = self.headers.get("Content-Type", "audio/webm").split(";")[0].strip()
            ext_map = {"audio/webm":".webm","audio/ogg":".ogg","audio/mp4":".mp4","audio/mpeg":".mp3"}
            suffix = ext_map.get(ctype, ".webm")

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            try:
                segments = transcribe_moonshine_ja(tmp_path)
                self._json_response(200, {"segments": segments})
            finally:
                try: os.unlink(tmp_path)
                except OSError: pass

        except Exception as e:
            self._json_response(500, {"error": f"Moonshine 文字起こしエラー: {e}"})

    # ─── /api/analyze (Azure OpenAI) ───────
    def _proxy_openai(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length)) if length else {}

            api_key     = body.pop("azure_api_key", "").strip()
            endpoint    = body.pop("azure_endpoint", "").strip().rstrip("/")
            deployment  = body.pop("azure_deployment", "").strip()
            api_version = body.pop("azure_api_version", "2025-01-01-preview").strip()

            if not api_key or not endpoint or not deployment:
                self._json_response(400, {
                    "error": "Azure API キー・エンドポイント・デプロイ名が必要です。⚙ から設定してください。"
                })
                return

            try:
                from openai import AzureOpenAI
            except ImportError:
                self._json_response(500, {
                    "error": "openai パッケージが未インストールです。pip install openai を実行してください。"
                })
                return

            system     = body.get("system", "")
            messages   = list(body.get("messages", []))
            max_tokens = body.get("max_tokens", 4096)

            if system:
                messages = [{"role": "system", "content": system}] + messages

            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
            )
            resp = client.chat.completions.create(
                model=deployment,
                messages=messages,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""

            # Anthropic 互換形式で返す（ブラウザ側の解析コードを流用）
            self._json_response(200, {
                "content": [{"type": "text", "text": text}]
            })

        except Exception as e:
            self._json_response(500, {"error": f"Azure OpenAI エラー: {e}"})

    # ─── 共通ユーティリティ ───────────────
    def _json_response(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type",   "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type, X-HF-Token, X-Diarize-Model, X-Session-ID, X-Vocab")
        self.end_headers()

    def log_message(self, format, *args):
        first = str(args[0]) if args else ""
        if "/api/" in first:
            sys.stderr.write(f"  API: {first}\n")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = http.server.HTTPServer(("0.0.0.0", PORT), SecondBrainHandler)
    print(f"""
╔══════════════════════════════════════════════════╗
║  🧠 セカンドブレーン                              ║
║                                                  ║
║  → http://localhost:{PORT}                          ║
║                                                  ║
║  エンジン:                                        ║
║    WhisperX  → /api/transcribe                   ║
║    Moonshine JA → /api/moonshine-transcribe      ║
║  分析:                                            ║
║    Azure OpenAI → /api/analyze                   ║
║                                                  ║
║  Ctrl+C で停止                                    ║
╚══════════════════════════════════════════════════╝
""")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n停止しました。")
        server.server_close()


if __name__ == "__main__":
    main()
