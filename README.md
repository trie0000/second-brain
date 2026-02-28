# 🧠 セカンドブレーン

会議音声をリアルタイムで文字起こしし、AIが論点・決定事項・アクションアイテムをその場で構造化・可視化するWebツール。

## 概要

- **文字起こし**: WhisperX（ローカルOSS）で15秒チャンクごとに高精度STT
- **話者分離**: pyannoteによる話者識別（HuggingFace Token が必要）
- **論点分析**: Claude API（Sonnet）が議題・論点・決定事項・アクションアイテムを構造化
- **ブラウザ完結**: Bot参加・インストール不要。Teams/Zoom/Meetなどあらゆる会議ツールで利用可

## 必要なもの

| ツール | 用途 | 入手先 |
|--------|------|--------|
| Python 3.9+ | サーバー実行 | [python.org](https://www.python.org) |
| whisperx | 音声文字起こし | `pip install whisperx` |
| ffmpeg | 音声フォーマット変換 | `brew install ffmpeg` |
| Anthropic API キー | Claude分析 | [console.anthropic.com](https://console.anthropic.com) |
| HuggingFace Token | 話者分離（オプション） | [huggingface.co](https://huggingface.co/settings/tokens) |

> HuggingFace Tokenを使う場合は [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) の利用規約への同意が必要です。

## セットアップ

```bash
# 依存インストール
pip install whisperx
brew install ffmpeg   # macOS

# サーバー起動
python server.py
```

ブラウザで http://localhost:8080 を開き、画面右上 ⚙ から Anthropic APIキーを設定してください。

## 使い方

1. **録音開始** ボタンを押す
2. 会議音声（PCスピーカーからマイクで収音）を流す
3. 15秒ごとに自動で文字起こし → 200文字ごとにClaude分析が走る
4. 左ペイン: 話者ラベル付き文字起こし
5. 中央ペイン: 現在の議題・要約・論点
6. 右ペイン: 決定事項・アクションアイテム・分析履歴
7. 終了後は **コピー** でMarkdown形式のメモをクリップボードに取得

## コスト目安

| 内容 | 費用 |
|------|------|
| 文字起こし（WhisperX） | 無料（ローカル実行） |
| 1回のClaude分析 | 約3円 |
| 1時間会議（20回分析） | 約60円 |

## ファイル構成

```
second-brain/
├── index.html   # フロントエンド（UI・MediaRecorder・Claude分析）
└── server.py    # ローカルサーバー（WhisperX代理・Anthropic APIプロキシ）
```

## セキュリティ

- APIキー・HuggingFace Tokenはブラウザの`localStorage`に保存
- サーバーは`localhost`のみで動作（外部公開なし）
- Anthropic APIへのリクエストはサーバー経由（CORSバイパスのため）
