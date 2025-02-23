import asyncio
import base64
import json
import os
import sys
import threading
from typing import Any

import numpy as np
import sounddevice as sd
import websocket
from dotenv import load_dotenv

load_dotenv()

REALTIME_API_URL = (
    "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# グローバル変数としてWebSocketを保持
global_ws: websocket.WebSocketApp | None = None

SYSTEM_PROMPT = f"""
あなたは日本語でのみ会話してください。
"""


def audio_callback(
    indata: np.ndarray[Any, np.dtype[np.int16]],
    frames: int,
    time: Any,
    status: sd.CallbackFlags,
) -> None:
    # 無音なら送信しない（必要に応じてしきい値を調整）
    amplitude = np.abs(indata).mean()
    if amplitude < 10:
        return

    # 音声データをBase64エンコードし、サーバーに送信
    audio_chunk = base64.b64encode(indata.tobytes()).decode("utf-8")
    if global_ws and global_ws.sock and global_ws.sock.connected:
        payload = json.dumps(
            {"type": "input_audio_buffer.append", "audio": audio_chunk}
        )
        global_ws.send(payload)


def on_open(ws: websocket.WebSocket) -> None:
    # WSを介してクライアントからサーバーへイベントをsession.updateのイベントを送信する
    # サーバーのデフォルト設定からこちらの希望する設定に変更することが可能
    init_payload = json.dumps(
        {
            "type": "session.update",
            "session": {
                "instructions": SYSTEM_PROMPT,
            },
        }
    )
    ws.send(init_payload)


def on_message(ws: websocket.WebSocket, message: str) -> None:
    try:
        response = json.loads(message)
        if "type" in response and response["type"] == "session.created":
            # WSを介してセッションを開始した時にサーバー側で送信されるイベント
            # デフォルトの設定等が入っている
            print("✅ セッションが作成されました")
            print(f"✅ セッション開始ログ: {response}")
        if "type" in response and response["type"] == "session.updated":
            print("✅ セッションが更新されました")
            print(f"✅ セッション更新ログ: {response}")
        if "type" in response and response["type"] == "response.audio_transcript.delta":
            if "delta" in response:
                sys.stdout.write(response["delta"])
                sys.stdout.flush()
        if "type" in response and response["type"] == "response.audio_transcript.done":
            if "transcript" in response:
                # transcriptの中に最終のテキスト全体が入っている
                # print(f"最終結果: {response['transcript']}")
                print("\n✅ リアルタイム録音中... Ctrl+C で停止")
    except json.JSONDecodeError:
        print("JSON decode error, message:", message)


def on_error(ws: websocket.WebSocket, error: str) -> None:
    print(f"❌ WebSocket エラー: {error}")


def run_ws() -> None:
    global global_ws
    headers = [f"Authorization: Bearer {OPENAI_API_KEY}", "OpenAI-Beta: realtime=v1"]

    global_ws = websocket.WebSocketApp(
        REALTIME_API_URL,
        header=headers,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
    )
    global_ws.run_forever()


async def main() -> None:
    # WebSocketを別スレッドで実行
    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()

    # マイクからの音声入力を開始
    with sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype=np.int16,
        blocksize=1024,
        callback=audio_callback,
    ):
        print("✅ リアルタイム録音中... Ctrl+C で停止")
        while True:
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
