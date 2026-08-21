#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdio.h>

typedef struct lua_State lua_State;
typedef int (__cdecl *lua_Writer)(lua_State *, const void *, size_t, void *);
typedef lua_State *(__cdecl *luaL_newstate_fn)(void);
typedef int (__cdecl *luaL_loadfile_fn)(lua_State *, const char *);
typedef int (__cdecl *lua_dump_fn)(lua_State *, lua_Writer, void *);
typedef const char *(__cdecl *lua_tolstring_fn)(lua_State *, int, size_t *);
typedef void (__cdecl *lua_close_fn)(lua_State *);

static int __cdecl write_chunk(lua_State *state, const void *data, size_t size, void *context) {
    (void)state;
    return fwrite(data, 1, size, (FILE *)context) == size ? 0 : 1;
}

int main(int argc, char **argv) {
    HMODULE library;
    lua_State *state;
    FILE *output;
    luaL_newstate_fn luaL_newstate_ptr;
    luaL_loadfile_fn luaL_loadfile_ptr;
    lua_dump_fn lua_dump_ptr;
    lua_tolstring_fn lua_tolstring_ptr;
    lua_close_fn lua_close_ptr;
    int result;

    if (argc != 4) {
        fprintf(stderr, "usage: compile_lua <lua51.dll> <input.lua> <output.luac>\n");
        return 2;
    }
    library = LoadLibraryA(argv[1]);
    if (library == NULL) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 3;
    }
#define RESOLVE(name) name##_ptr = (name##_fn)GetProcAddress(library, #name)
    RESOLVE(luaL_newstate);
    RESOLVE(luaL_loadfile);
    RESOLVE(lua_dump);
    RESOLVE(lua_tolstring);
    RESOLVE(lua_close);
#undef RESOLVE
    if (!luaL_newstate_ptr || !luaL_loadfile_ptr || !lua_dump_ptr
        || !lua_tolstring_ptr || !lua_close_ptr) {
        fprintf(stderr, "lua51.dll is missing a required compiler export\n");
        return 4;
    }
    state = luaL_newstate_ptr();
    if (state == NULL) {
        return 5;
    }
    result = luaL_loadfile_ptr(state, argv[2]);
    if (result != 0) {
        fprintf(stderr, "%s\n", lua_tolstring_ptr(state, -1, NULL));
        lua_close_ptr(state);
        return 6;
    }
    if (fopen_s(&output, argv[3], "wb") != 0) {
        lua_close_ptr(state);
        return 7;
    }
    result = lua_dump_ptr(state, write_chunk, output);
    fclose(output);
    lua_close_ptr(state);
    return result == 0 ? 0 : 8;
}
