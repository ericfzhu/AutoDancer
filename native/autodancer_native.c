#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct lua_State lua_State;
typedef int (__cdecl *lua_CFunction)(lua_State *state);

typedef void (__cdecl *lua_pushnil_fn)(lua_State *state);
typedef void (__cdecl *lua_pushboolean_fn)(lua_State *state, int value);
typedef void (__cdecl *lua_pushlstring_fn)(lua_State *state, const char *value, size_t length);
typedef void (__cdecl *lua_createtable_fn)(lua_State *state, int array_size, int record_size);
typedef const char *(__cdecl *luaL_checklstring_fn)(lua_State *state, int index, size_t *length);
typedef void (__cdecl *luaL_register_fn)(lua_State *state, const char *name, const void *functions);

typedef struct luaL_Reg {
    const char *name;
    lua_CFunction function;
} luaL_Reg;

static lua_pushnil_fn lua_pushnil_ptr;
static lua_pushboolean_fn lua_pushboolean_ptr;
static lua_pushlstring_fn lua_pushlstring_ptr;
static lua_createtable_fn lua_createtable_ptr;
static luaL_checklstring_fn luaL_checklstring_ptr;
static luaL_register_fn luaL_register_ptr;
static HANDLE pipe_handle = INVALID_HANDLE_VALUE;

static int resolve_lua_api(void) {
    HMODULE lua = GetModuleHandleW(L"lua51.dll");
    if (lua == NULL) {
        return 0;
    }
#define RESOLVE(name) name##_ptr = (name##_fn)GetProcAddress(lua, #name)
    RESOLVE(lua_pushnil);
    RESOLVE(lua_pushboolean);
    RESOLVE(lua_pushlstring);
    RESOLVE(lua_createtable);
    RESOLVE(luaL_checklstring);
    RESOLVE(luaL_register);
#undef RESOLVE
    return lua_pushnil_ptr && lua_pushboolean_ptr && lua_pushlstring_ptr && lua_createtable_ptr
        && luaL_checklstring_ptr && luaL_register_ptr;
}

static int connect_pipe(void) {
    wchar_t pipe_name[256];
    DWORD length;
    DWORD mode = PIPE_READMODE_MESSAGE;

    if (pipe_handle != INVALID_HANDLE_VALUE) {
        return 1;
    }
    length = GetEnvironmentVariableW(L"AUTODANCER_PIPE", pipe_name, 256);
    if (length == 0 || length >= 256) {
        return 0;
    }
    pipe_handle = CreateFileW(
        pipe_name,
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    if (pipe_handle == INVALID_HANDLE_VALUE) {
        return 0;
    }
    if (!SetNamedPipeHandleState(pipe_handle, &mode, NULL, NULL)) {
        CloseHandle(pipe_handle);
        pipe_handle = INVALID_HANDLE_VALUE;
        return 0;
    }
    return 1;
}

__declspec(dllexport) const char *__cdecl autodancer_get_instance_id(void) {
    static char instance_id[128];
    DWORD length = GetEnvironmentVariableA("AUTODANCER_INSTANCE_ID", instance_id, 128);
    if (length == 0 || length >= 128) {
        return "worker-unknown";
    }
    return instance_id;
}

__declspec(dllexport) int __cdecl autodancer_poll(char *message, int capacity) {
    DWORD available = 0;
    DWORD received = 0;

    if (message == NULL || capacity <= 0 || !connect_pipe()
        || !PeekNamedPipe(pipe_handle, NULL, 0, NULL, &available, NULL) || available == 0) {
        return 0;
    }
    if (available > (DWORD)capacity) {
        available = (DWORD)capacity;
    }
    if (!ReadFile(pipe_handle, message, available, &received, NULL) || received == 0) {
        CloseHandle(pipe_handle);
        pipe_handle = INVALID_HANDLE_VALUE;
        return 0;
    }
    return (int)received;
}

__declspec(dllexport) int __cdecl autodancer_send(const char *message, int length) {
    DWORD sent = 0;
    return message != NULL && length > 0 && connect_pipe()
        && WriteFile(pipe_handle, message, (DWORD)length, &sent, NULL)
        && sent == (DWORD)length;
}

__declspec(dllexport) void __cdecl autodancer_close(void) {
    if (pipe_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(pipe_handle);
        pipe_handle = INVALID_HANDLE_VALUE;
    }
}

static int native_get_instance_id(lua_State *state) {
    const char *instance_id = autodancer_get_instance_id();
    lua_pushlstring_ptr(state, instance_id, strlen(instance_id));
    return 1;
}

static int native_poll(lua_State *state) {
    char message[4096];
    DWORD available = 0;
    DWORD received = 0;

    if (!connect_pipe() || !PeekNamedPipe(pipe_handle, NULL, 0, NULL, &available, NULL)
        || available == 0) {
        lua_pushnil_ptr(state);
        return 1;
    }
    if (available > sizeof(message)) {
        available = sizeof(message);
    }
    if (!ReadFile(pipe_handle, message, available, &received, NULL) || received == 0) {
        CloseHandle(pipe_handle);
        pipe_handle = INVALID_HANDLE_VALUE;
        lua_pushnil_ptr(state);
        return 1;
    }
    lua_pushlstring_ptr(state, message, received);
    return 1;
}

static int native_send(lua_State *state) {
    size_t length = 0;
    DWORD sent = 0;
    const char *message = luaL_checklstring_ptr(state, 1, &length);
    int ok = connect_pipe() && length <= UINT32_MAX
        && WriteFile(pipe_handle, message, (DWORD)length, &sent, NULL)
        && sent == length;
    lua_pushboolean_ptr(state, ok);
    return 1;
}

static int native_close(lua_State *state) {
    (void)state;
    autodancer_close();
    return 0;
}

__declspec(dllexport) int __cdecl luaopen_autodancer_native(lua_State *state) {
    static const luaL_Reg functions[] = {
        {"getInstanceID", native_get_instance_id},
        {"poll", native_poll},
        {"send", native_send},
        {"close", native_close},
        {NULL, NULL},
    };
    if (!resolve_lua_api()) {
        return 0;
    }
    lua_createtable_ptr(state, 0, 4);
    luaL_register_ptr(state, NULL, functions);
    return 1;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved) {
    (void)instance;
    (void)reserved;
    if (reason == DLL_PROCESS_DETACH && pipe_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(pipe_handle);
        pipe_handle = INVALID_HANDLE_VALUE;
    }
    return TRUE;
}
