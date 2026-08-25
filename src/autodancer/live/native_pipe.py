"""Windows named-pipe server used by the in-process Lua bridge module."""

from __future__ import annotations

import ctypes
import os
import threading
import time
from ctypes import wintypes


class NativePipeError(RuntimeError):
    """Raised when the native transport cannot be created or used."""


MAX_PIPE_MESSAGE_BYTES = 262144


if os.name == "nt":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

    class _SecurityAttributes(ctypes.Structure):
        _fields_ = [
            ("nLength", wintypes.DWORD),
            ("lpSecurityDescriptor", wintypes.LPVOID),
            ("bInheritHandle", wintypes.BOOL),
        ]

    class _Overlapped(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    _kernel32.CreateNamedPipeW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_SecurityAttributes),
    ]
    _kernel32.CreateNamedPipeW.restype = wintypes.HANDLE
    _kernel32.ConnectNamedPipe.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Overlapped)]
    _kernel32.ConnectNamedPipe.restype = wintypes.BOOL
    _kernel32.WriteFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPCVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    _kernel32.WriteFile.restype = wintypes.BOOL
    _kernel32.ReadFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    _kernel32.ReadFile.restype = wintypes.BOOL
    _kernel32.PeekNamedPipe.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    _kernel32.PeekNamedPipe.restype = wintypes.BOOL
    _kernel32.CancelIoEx.argtypes = [wintypes.HANDLE, wintypes.LPVOID]
    _kernel32.GetOverlappedResult.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_Overlapped),
        ctypes.POINTER(wintypes.DWORD),
        wintypes.BOOL,
    ]
    _kernel32.GetOverlappedResult.restype = wintypes.BOOL
    _kernel32.CreateEventW.argtypes = [
        wintypes.LPVOID,
        wintypes.BOOL,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    _kernel32.CreateEventW.restype = wintypes.HANDLE
    _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD
    _kernel32.DisconnectNamedPipe.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.DWORD),
    ]
    _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL

    _INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
    _PIPE_ACCESS_DUPLEX = 0x00000003
    _FILE_FLAG_FIRST_PIPE_INSTANCE = 0x00080000
    _FILE_FLAG_OVERLAPPED = 0x40000000
    _PIPE_TYPE_MESSAGE = 0x00000004
    _PIPE_READMODE_MESSAGE = 0x00000002
    _PIPE_WAIT = 0x00000000
    _PIPE_UNLIMITED_INSTANCES = 255
    _ERROR_PIPE_CONNECTED = 535
    _ERROR_IO_PENDING = 997
    _WAIT_OBJECT_0 = 0
    _WAIT_TIMEOUT = 258
    _PIPE_BUFFER_BYTES = MAX_PIPE_MESSAGE_BYTES


def _milliseconds(timeout: float) -> int:
    return max(1, min(int(timeout * 1000), 0xFFFFFFFE))


def _new_overlapped() -> tuple[_Overlapped, int]:
    event = _kernel32.CreateEventW(None, True, False, None)
    if not event:
        raise NativePipeError(f"CreateEventW failed: {ctypes.get_last_error()}")
    overlapped = _Overlapped()
    overlapped.hEvent = event
    return overlapped, event


def _finish_overlapped(
    handle: int,
    overlapped: _Overlapped,
    event: int,
    timeout: float,
    operation: str,
) -> int:
    result = _kernel32.WaitForSingleObject(event, _milliseconds(timeout))
    if result == _WAIT_TIMEOUT:
        _kernel32.CancelIoEx(handle, ctypes.byref(overlapped))
        raise NativePipeError(f"{operation} timed out after {timeout:.1f} seconds")
    if result != _WAIT_OBJECT_0:
        raise NativePipeError(f"{operation} wait failed: {ctypes.get_last_error()}")
    transferred = wintypes.DWORD()
    if not _kernel32.GetOverlappedResult(
        handle, ctypes.byref(overlapped), ctypes.byref(transferred), False
    ):
        raise NativePipeError(f"{operation} failed: {ctypes.get_last_error()}")
    return int(transferred.value)


def pipe_name(session_id: str, instance_id: str) -> str:
    for value in (session_id, instance_id):
        if not value or not all(character.isalnum() or character in "-_" for character in value):
            raise ValueError("pipe identifiers may contain only letters, digits, '-' and '_'")
    return rf"\\.\pipe\AutoDancer-{session_id}-{instance_id}"


class NativePipeServer:
    """Own one current-user-only duplex message pipe for a game process."""

    def __init__(self, name: str) -> None:
        if os.name != "nt":
            raise NativePipeError("the live native bridge is supported only on Windows")
        self.name = name
        self._connected = threading.Event()
        self._closed = False
        self._write_lock = threading.Lock()
        self._read_lock = threading.Lock()
        descriptor = wintypes.LPVOID()
        # Protected DACL: LocalSystem and the object owner (the current user).
        if not _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            "D:P(A;;GA;;;SY)(A;;GA;;;OW)", 1, ctypes.byref(descriptor), None
        ):
            raise NativePipeError(f"security descriptor failed: {ctypes.get_last_error()}")
        attributes = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), descriptor, False)
        try:
            self._handle = _kernel32.CreateNamedPipeW(
                name,
                _PIPE_ACCESS_DUPLEX | _FILE_FLAG_FIRST_PIPE_INSTANCE | _FILE_FLAG_OVERLAPPED,
                _PIPE_TYPE_MESSAGE | _PIPE_READMODE_MESSAGE | _PIPE_WAIT,
                _PIPE_UNLIMITED_INSTANCES,
                _PIPE_BUFFER_BYTES,
                _PIPE_BUFFER_BYTES,
                0,
                ctypes.byref(attributes),
            )
        finally:
            _kernel32.LocalFree(descriptor)
        if self._handle == _INVALID_HANDLE_VALUE:
            raise NativePipeError(f"CreateNamedPipeW failed: {ctypes.get_last_error()}")
        self._connect_thread = threading.Thread(
            target=self._connect, name=f"pipe:{name}", daemon=True
        )
        self._connect_thread.start()

    def _connect(self) -> None:
        overlapped, event = _new_overlapped()
        try:
            ok = _kernel32.ConnectNamedPipe(self._handle, ctypes.byref(overlapped))
            error = ctypes.get_last_error()
            if ok or error == _ERROR_PIPE_CONNECTED:
                self._connected.set()
            elif error == _ERROR_IO_PENDING:
                result = _kernel32.WaitForSingleObject(event, 0xFFFFFFFF)
                if result == _WAIT_OBJECT_0 and not self._closed:
                    self._connected.set()
        finally:
            _kernel32.CloseHandle(event)

    def send(self, payload: bytes, timeout: float = 10.0) -> None:
        if self._closed:
            raise NativePipeError("named pipe is closed")
        if not payload or len(payload) > 4096:
            raise ValueError("pipe messages must contain 1..4096 bytes")
        if not self._connected.wait(timeout):
            raise NativePipeError(f"timed out waiting for game pipe {self.name}")
        buffer = ctypes.create_string_buffer(payload)
        with self._write_lock:
            overlapped, event = _new_overlapped()
            try:
                sent = wintypes.DWORD()
                ok = _kernel32.WriteFile(
                    self._handle,
                    buffer,
                    len(payload),
                    ctypes.byref(sent),
                    ctypes.byref(overlapped),
                )
                if ok:
                    count = int(sent.value)
                elif ctypes.get_last_error() == _ERROR_IO_PENDING:
                    count = _finish_overlapped(
                        self._handle, overlapped, event, timeout, "named-pipe write"
                    )
                else:
                    raise NativePipeError(f"WriteFile failed: {ctypes.get_last_error()}")
            finally:
                _kernel32.CloseHandle(event)
        if count != len(payload):
            raise NativePipeError(f"WriteFile sent {count} of {len(payload)} bytes")

    def receive(
        self, timeout: float = 10.0, *, max_bytes: int = MAX_PIPE_MESSAGE_BYTES
    ) -> bytes:
        """Receive one complete game-to-Python message without polling the filesystem."""
        if self._closed:
            raise NativePipeError("named pipe is closed")
        if not 0 < max_bytes <= MAX_PIPE_MESSAGE_BYTES:
            raise ValueError(f"max_bytes must be in 1..{MAX_PIPE_MESSAGE_BYTES}")
        if not self._connected.wait(timeout):
            raise NativePipeError(f"timed out waiting for game pipe {self.name}")
        deadline = time.monotonic() + timeout
        with self._read_lock:
            while True:
                available = wintypes.DWORD()
                message_bytes = wintypes.DWORD()
                ok = _kernel32.PeekNamedPipe(
                    self._handle,
                    None,
                    0,
                    None,
                    ctypes.byref(available),
                    ctypes.byref(message_bytes),
                )
                if not ok:
                    raise NativePipeError(f"PeekNamedPipe failed: {ctypes.get_last_error()}")
                if message_bytes.value:
                    if message_bytes.value > max_bytes:
                        raise NativePipeError(
                            f"game pipe message is {message_bytes.value} bytes; "
                            f"limit is {max_bytes}"
                        )
                    buffer = ctypes.create_string_buffer(message_bytes.value)
                    overlapped, event = _new_overlapped()
                    try:
                        received = wintypes.DWORD()
                        ok = _kernel32.ReadFile(
                            self._handle,
                            buffer,
                            message_bytes.value,
                            ctypes.byref(received),
                            ctypes.byref(overlapped),
                        )
                        if ok:
                            count = int(received.value)
                        elif ctypes.get_last_error() == _ERROR_IO_PENDING:
                            count = _finish_overlapped(
                                self._handle,
                                overlapped,
                                event,
                                max(deadline - time.monotonic(), 0.001),
                                "named-pipe read",
                            )
                        else:
                            raise NativePipeError(
                                f"ReadFile failed: {ctypes.get_last_error()}"
                            )
                    finally:
                        _kernel32.CloseHandle(event)
                    if count != message_bytes.value:
                        raise NativePipeError(
                            f"ReadFile received {count} of {message_bytes.value} bytes"
                        )
                    return buffer.raw[:count]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"No AutoDancer pipe record arrived within {timeout:.1f} seconds"
                    )
                time.sleep(min(0.001, remaining))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _kernel32.CancelIoEx(self._handle, None)
        _kernel32.DisconnectNamedPipe(self._handle)
        _kernel32.CloseHandle(self._handle)

    def __enter__(self) -> NativePipeServer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
