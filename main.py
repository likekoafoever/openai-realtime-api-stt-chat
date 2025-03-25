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

# 글로벌변수로해서 WebSocket
global_ws: websocket.WebSocketApp | None = None

SYSTEM_PROMPT = f"""
당신은 한국어로 대화해 주세요.
"""


def audio_callback(
    indata: np.ndarray[Any, np.dtype[np.int16]],
    frames: int,
    time: Any,
    status: sd.CallbackFlags,
) -> None:
    # 무음은 송신하지 않는다 
    amplitude = np.abs(indata).mean()
    if amplitude < 10:
        return

    # 음성데이터를 Base64인코드해서 서버로 송신
    audio_chunk = base64.b64encode(indata.tobytes()).decode("utf-8")
    if global_ws and global_ws.sock and global_ws.sock.connected:
        payload = json.dumps(
            {"type": "input_audio_buffer.append", "audio": audio_chunk}
        )
        global_ws.send(payload)


def on_open(ws: websocket.WebSocket) -> None:
    # WS를 열어서 클라이언트로부터 서버로 session.update의 이벤트를 송신한다.
    # 서버의 디폴트설정으로부터 희망하는 설정을 변경가능
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
            # WS를 열어 세션을 개시하는 동안 서버측에 송신하는 이벤트 
            print("✅ 세션이 시작되었습니다.")
            print(f"✅ 세션 개시 로그 {response}")
        if "type" in response and response["type"] == "session.updated":
            print("✅ 세션이 갱신되었습니다.")
            print(f"✅ 세션 갱신 로그 {response}")
        if "type" in response and response["type"] == "response.audio_transcript.delta":
            sys.stdout.write(response["delta"])
            sys.stdout.flush()
        if "type" in response and response["type"] == "response.audio_transcript.done":
            # transcript의 중간에 최종 텍스트 전체를 입력
            # print(f"최종결과: {response['transcript']}")
            print("\n✅ 리얼타임 녹음 중... Ctrl+C 로 중지")
    except json.JSONDecodeError:
        print("JSON decode error, message:", message)


def on_error(ws: websocket.WebSocket, error: str) -> None:
    print(f"❌ WebSocket 에러 : {error}")


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
    # WebSocket을 별도의 스레드로 실행
    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()

    # 마이크로부터 음성입력을 개시
    with sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype=np.int16,
        blocksize=1024,
        callback=audio_callback,
    ):
        print("✅ 리얼타임녹음중... Ctrl+C로 정지")
        while True:
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
