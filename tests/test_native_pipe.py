from __future__ import annotations

import ctypes
import os
import queue
import threading
import uuid
from ctypes import wintypes

import pytest

from autodancer.live.native_pipe import NativePipeError, NativePipeServer, pipe_name

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows named-pipe transport")


def _client(name: str, outbound: bytes, received: queue.Queue[bytes | BaseException]) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    handle = kernel32.CreateFileW(name, 0xC0000000, 0, None, 3, 0, None)
    try:
        sent = wintypes.DWORD()
        payload = ctypes.create_string_buffer(outbound)
        if not kernel32.WriteFile(handle, payload, len(outbound), ctypes.byref(sent), None):
            raise OSError(ctypes.get_last_error())
        buffer = ctypes.create_string_buffer(512)
        count = wintypes.DWORD()
        if not kernel32.ReadFile(handle, buffer, len(buffer), ctypes.byref(count), None):
            raise OSError(ctypes.get_last_error())
        received.put(buffer.raw[: count.value])
    except BaseException as error:
        received.put(error)
    finally:
        kernel32.CloseHandle(handle)


def test_native_pipe_is_duplex_and_accepts_messages_larger_than_4k() -> None:
    server = NativePipeServer(pipe_name(uuid.uuid4().hex, "worker-0000"))
    replies: queue.Queue[bytes | BaseException] = queue.Queue()
    payload = b"x" * 20_000
    client = threading.Thread(target=_client, args=(server.name, payload, replies))
    client.start()
    try:
        assert server.receive(2) == payload
        server.send(b"ACTION session 1 0\n", 2)
        reply = replies.get(timeout=2)
        if isinstance(reply, BaseException):
            raise reply
        assert reply == b"ACTION session 1 0\n"
    finally:
        client.join(timeout=2)
        server.close()


def test_native_pipe_connection_timeout_is_bounded() -> None:
    server = NativePipeServer(pipe_name(uuid.uuid4().hex, "worker-0000"))
    try:
        with pytest.raises(NativePipeError, match="timed out waiting"):
            server.receive(0.01)
    finally:
        server.close()
