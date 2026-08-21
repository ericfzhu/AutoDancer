-- Trusted, build-pinned loader shim for the AutoDancer transport DLL.

local ffi = ...

ffi.cdef [[
const char *autodancer_get_instance_id(void);
int autodancer_poll(char *message, int capacity);
int autodancer_send(const char *message, int length);
void autodancer_close(void);
]]

local native = ffi.load("autodancer_native")
local buffer = ffi.new("char[4096]")
local module = {}

function module.getInstanceID()
    return ffi.string(native.autodancer_get_instance_id())
end

function module.poll()
    local length = native.autodancer_poll(buffer, 4096)
    if length > 0 then
        return ffi.string(buffer, length)
    end
end

function module.send(message)
    return native.autodancer_send(message, #message) ~= 0
end

function module.close()
    native.autodancer_close()
end

return module
