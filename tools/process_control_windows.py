"""Small Windows process-handle helpers for the orchestration-only handoff."""
import ctypes
from ctypes import wintypes
import os


class WinProcess:
    def __init__(self, pid, expected_creation=None):
        if os.name != "nt":
            raise RuntimeError("Windows process handles required")
        self.pid = int(pid)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        self.kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.handle = self.kernel.OpenProcess(0x100000 | 0x1000 | 0x800 | 1, False, self.pid)
        if not self.handle:
            code = ctypes.get_last_error()
            if code == 87:
                raise ProcessLookupError(self.pid)
            raise ctypes.WinError(code)
        values = [wintypes.FILETIME() for _ in range(4)]
        if not self.kernel.GetProcessTimes(self.handle, *[ctypes.byref(x) for x in values]):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())
        self.creation_time_100ns = (values[0].dwHighDateTime << 32) | values[0].dwLowDateTime
        if expected_creation is not None and self.creation_time_100ns != int(expected_creation):
            self.close()
            raise ProcessLookupError("PID was reused: " + str(self.pid))

    def exit_code(self):
        value = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        return None if value.value == 259 else value.value

    def is_running(self):
        return self.exit_code() is None

    def suspend(self):
        fn = ctypes.WinDLL("ntdll").NtSuspendProcess
        fn.argtypes = [wintypes.HANDLE]
        fn.restype = wintypes.LONG
        status = fn(self.handle)
        if status < 0:
            raise OSError("NtSuspendProcess failed: " + str(status))

    def resume(self):
        fn = ctypes.WinDLL("ntdll").NtResumeProcess
        fn.argtypes = [wintypes.HANDLE]
        fn.restype = wintypes.LONG
        status = fn(self.handle)
        if status < 0:
            raise OSError("NtResumeProcess failed: " + str(status))

    def terminate(self, code=173):
        if not self.kernel.TerminateProcess(self.handle, code):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if getattr(self, "handle", None):
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def split_command_line(command):
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    size = ctypes.c_int()
    args = shell.CommandLineToArgvW(command, ctypes.byref(size))
    if not args:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return [args[i] for i in range(size.value)]
    finally:
        free = ctypes.WinDLL("kernel32").LocalFree
        free.argtypes = [wintypes.HLOCAL]
        free(args)
