#!/bin/bash
# セカンドブレーン 起動スクリプト
# 使い方: ./run.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/venv"

# venv がなければ作成
if [ ! -d "$VENV" ]; then
    echo "venvが見つかりません。python3.10 で作成します..."
    python3.10 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install whisperx
fi

# venv の Python でサーバー起動
exec "$VENV/bin/python" "$SCRIPT_DIR/server.py"
