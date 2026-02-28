#!/usr/bin/env python3
"""
セカンドブレーン - ローカルサーバー（WhisperX版）

依存: pip install whisperx  /  brew install ffmpeg
話者分離を使う場合は HuggingFace Token も必要

使い方:
  python server.py
  → ブラウザで http://localhost:8080 を開く
  → 画面右上の ⚙ からAPIキー・HFトークンを設定
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
# WhisperX モデル管理（グローバル・遅延ロード）
# ──────────────────────────────────────────
_whisper_model = None
_diarize_model = None
_model_lock = threading.Lock()
DEVICE = "cpu"
COMPUTE_TYPE = "int8"   # CPUで動くfloat32より速い


def _get_whisper():
    """Whisperモデルを取得（初回のみロード）"""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisperx
        except ImportError:
            raise RuntimeError(
                "whisperx がインストールされていません。\n"
                "  pip install whisperx\n"
                "を実行してから再起動してください。"
            )
        print("  [WhisperX] Whisperモデルを読み込み中 (base, ja)...", flush=True)
        _whisper_model = whisperx.load_model(
            "base", DEVICE, compute_type=COMPUTE_TYPE, language="ja"
        )
        print("  [WhisperX] Whisperモデル完了", flush=True)
    return _whisper_model


def _get_diarize(hf_token):
    """話者分離モデルを取得（HFトークンが必要・初回のみロード）"""
    global _diarize_model
    if not hf_token:
        return None
    if _diarize_model is None:
        from whisperx.diarize import DiarizationPipeline
        print("  [WhisperX] 話者分離モデルを読み込み中...", flush=True)
        _diarize_model = DiarizationPipeline(
            token=hf_token, device=DEVICE
        )
        print("  [WhisperX] 話者分離モデル完了", flush=True)
    return _diarize_model


def transcribe_audio(audio_path, hf_token=""):
    """
    音声ファイルを文字起こし＋話者分離して返す。
    Returns: list of {"speaker": "SPEAKER_00", "text": "..."}
    """
    import whisperx

    with _model_lock:
        whisper_model = _get_whisper()
        diarize_model = _get_diarize(hf_token)

    audio = whisperx.load_audio(audio_path)
    result = whisper_model.transcribe(audio, batch_size=4)

    if not result.get("segments"):
        return []

    if diarize_model is not None:
        # 話者分離あり
        diarize_segs = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segs, result)
        segments_out = [
            {
                "speaker": seg.get("speaker", "SPEAKER_00"),
                "text": seg.get("text", "").strip(),
            }
            for seg in result.get("segments", [])
            if seg.get("text", "").strip()
        ]
    else:
        # 話者分離なし（全体を SPEAKER_00 として返す）
        full_text = " ".join(
            s.get("text", "").strip() for s in result.get("segments", [])
        ).strip()
        segments_out = [{"speaker": "SPEAKER_00", "text": full_text}] if full_text else []

    return segments_out


# ──────────────────────────────────────────
# HTTP ハンドラ
# ──────────────────────────────────────────
class SecondBrainHandler(http.server.SimpleHTTPRequestHandler):
    """静的ファイル配信 + WhisperX 文字起こし + Anthropic API プロキシ"""

    def do_POST(self):
        if self.path == "/api/transcribe":
            self._handle_transcribe()
        elif self.path == "/api/analyze":
            self._proxy_anthropic()
        else:
            self.send_error(404)

    # ── /api/transcribe ──────────────────
    def _handle_transcribe(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            hf_token = self.headers.get("X-HF-Token", "").strip()
            audio_bytes = self.rfile.read(length)

            # 音声をテンポラリファイルに保存（WebM形式）
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name

            try:
                segments = transcribe_audio(tmp_path, hf_token)
                self._json_response(200, {"segments": segments})
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        except Exception as e:
            self._json_response(500, {"error": f"文字起こしエラー: {str(e)}"})

    # ── /api/analyze ─────────────────────
    def _proxy_anthropic(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            api_key = body.pop("api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                self._json_response(400, {"error": "APIキーが設定されていません。画面右上の ⚙ から設定してください。"})
                return

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                self._json_response(200, json.loads(resp.read()))

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            try:
                msg = json.loads(error_body).get("error", {}).get("message", error_body)
            except json.JSONDecodeError:
                msg = error_body
            self._json_response(e.code, {"error": f"Anthropic API エラー ({e.code}): {msg}"})

        except urllib.error.URLError as e:
            self._json_response(502, {"error": f"API接続エラー: {e.reason}"})

        except Exception as e:
            self._json_response(500, {"error": f"サーバーエラー: {str(e)}"})

    # ── 共通ユーティリティ ─────────────
    def _json_response(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-HF-Token")
        self.end_headers()

    def log_message(self, format, *args):
        first = str(args[0]) if args else ""
        if "/api/" in first:
            sys.stderr.write(f"  API: {first}\n")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = http.server.HTTPServer(("localhost", PORT), SecondBrainHandler)
    print(f"""
╔══════════════════════════════════════════╗
║  🧠 セカンドブレーン - WhisperX版        ║
║                                          ║
║  → http://localhost:{PORT}                 ║
║                                          ║
║  Ctrl+C で停止                           ║
╚══════════════════════════════════════════╝
""")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n停止しました。")
        server.server_close()


if __name__ == "__main__":
    main()
