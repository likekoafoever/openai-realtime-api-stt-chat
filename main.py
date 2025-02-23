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

SAMPLE_RATE: int = 16000
CHANNELS: int = 1
CHUNK_SIZE: int = 1024

ws: websocket.WebSocketApp | None = None
loop: asyncio.AbstractEventLoop | None = None  # メインスレッドのイベントループ
connection_established = asyncio.Event()  # 接続完了を待つイベント


def audio_callback(
    indata: np.ndarray[Any, np.dtype[np.int16]],
    frames: int,
    time: Any,
    status: sd.CallbackFlags,
) -> None:
    """マイク音声をリアルタイムで WebSocket に送信"""
    # 無音なら送信しない（必要に応じてしきい値を調整）
    amplitude = np.abs(indata).mean()
    if amplitude < 10:
        return

    # 音声データを Base64 エンコード
    audio_chunk = base64.b64encode(indata.tobytes()).decode("utf-8")

    # WebSocket に音声データを送信（非同期で実行）
    if loop is not None:
        asyncio.run_coroutine_threadsafe(send_audio(audio_chunk), loop)


async def send_audio(audio_chunk: str) -> None:
    """音声データを WebSocket に送信"""
    await connection_established.wait()  # WebSocket 接続が確立するまで待機
    if ws and ws.sock and ws.sock.connected:
        payload = json.dumps(
            {"type": "input_audio_buffer.append", "audio": audio_chunk}
        )
        ws.send(payload)


def on_message(ws: websocket.WebSocketApp, message: str) -> None:
    """API からの文字起こし結果を受信"""
    try:
        response = json.loads(message)
        if "type" in response and response["type"] == "response.audio_transcript.delta":
            if "delta" in response:
                sys.stdout.write(response["delta"])
                sys.stdout.flush()
        if "type" in response and response["type"] == "response.audio_transcript.done":
            if "transcript" in response:
                # transcriptの中に最終のテキスト全体が入っている
                # print(f"最終結果: {response['transcript']}")
                print("\n")
    except json.JSONDecodeError:
        print("JSON decode error, message:", message)


def on_open(ws: websocket.WebSocketApp) -> None:
    """WebSocket 接続時の処理"""
    init_payload = json.dumps(
        {
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm16",
                "instructions": "あなたは日本語でのみ会話してください。",
            },
        }
    )
    ws.send(init_payload)

    async def async_set_event() -> None:
        """接続成功イベントを非同期でセット"""
        connection_established.set()

    if loop is not None:
        asyncio.run_coroutine_threadsafe(async_set_event(), loop)


def on_error(ws: websocket.WebSocketApp, error: str) -> None:
    """WebSocket のエラーハンドリング"""
    print(f"❌ WebSocket エラー: {error}")


def run_ws() -> None:
    """WebSocket を実行（別スレッド）"""
    global ws
    headers = [f"Authorization: Bearer {OPENAI_API_KEY}", "OpenAI-Beta: realtime=v1"]

    ws = websocket.WebSocketApp(
        REALTIME_API_URL,
        header=headers,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
    )
    ws.run_forever()


async def main() -> None:
    global loop
    loop = asyncio.get_running_loop()  # メインスレッドのイベントループを取得

    # WebSocket を別スレッドで実行
    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()

    # WebSocket の接続が完了するまで待機
    await connection_established.wait()
    print("✅ WebSocket 接続完了")

    # マイクからの音声入力を開始
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.int16,
        blocksize=CHUNK_SIZE,
        callback=audio_callback,
    ):
        print("🎤 リアルタイム録音中... Ctrl+C で停止")
        while True:
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
