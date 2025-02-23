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


global_ws: websocket.WebSocketApp | None = None
global_loop: asyncio.AbstractEventLoop | None = None
ws_connection_established = asyncio.Event()


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
    if global_loop is not None:
        asyncio.run_coroutine_threadsafe(send_audio(audio_chunk), global_loop)


async def send_audio(audio_chunk: str) -> None:
    """音声データを WebSocket に送信"""
    await ws_connection_established.wait()  # WebSocket 接続が確立するまで待機
    if global_ws and global_ws.sock and global_ws.sock.connected:
        payload = json.dumps(
            {"type": "input_audio_buffer.append", "audio": audio_chunk}
        )
        global_ws.send(payload)


def on_open(ws: websocket.WebSocket) -> None:
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
        ws_connection_established.set()

    if global_loop is not None:
        asyncio.run_coroutine_threadsafe(async_set_event(), global_loop)


def on_message(ws: websocket.WebSocket, message: str) -> None:
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
                print("\n✅ リアルタイム録音中... Ctrl+C で停止")
    except json.JSONDecodeError:
        print("JSON decode error, message:", message)


def on_error(ws: websocket.WebSocket, error: str) -> None:
    print(f"❌ WebSocket エラー: {error}")


def run_ws() -> None:
    """WebSocket を実行（別スレッド）"""
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
    global global_loop
    global_loop = asyncio.get_running_loop()  # メインスレッドのイベントループを取得

    # WebSocket を別スレッドで実行
    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()

    # WebSocket の接続が完了するまで待機
    await ws_connection_established.wait()
    print("✅ WebSocket 接続完了")

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
