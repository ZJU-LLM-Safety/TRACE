import json
import os
import time
import docker
import struct
import socket
import re
import io
import tarfile
import subprocess
import logging

logger = logging.getLogger(__name__)

class Container:
    def __init__(self, container_name=None):
        self.client = docker.from_env()
        self.container = self._get_container(container_name)

        if not self.container:
            raise ValueError("Container not found!")

        # 启动一个持久的 Bash shell
        self.exec_id = self.client.api.exec_create(
            self.container.id, "bash -i", stdin=True, tty=True
        )["Id"]
        self.sock = self.client.api.exec_start(self.exec_id, socket=True)._sock
        self.sock.settimeout(5)

        # 清理初始的 shell 输出
        self._clear_initial_output()

    def _get_container(self, container_name):
        """获取正在运行的容器"""
        if container_name:
            try:
                return self.client.containers.get(container_name)
            except docker.errors.NotFound:
                return None
        containers = self.client.containers.list(filters={"label": "com.docker.compose.service"})
        return containers[0] if containers else None

    def __del__(self):
        try:
            self.container.stop()
        except:
            pass

    def _clear_initial_output(self):
        """清除启动 bash shell 时的初始输出"""
        try:
            time.sleep(0.2)  # 等待 shell 启动
            self.sock.recv(4096)  # 读取缓冲区内容
        except socket.timeout:
            pass


    def _send_command(self, command):
        self.sock.send(command.encode("utf-8") + b'\n')
        data = self.sock.recv(8)
        _, n = struct.unpack('>BxxxL', data)
        self.sock.recv(n)

    def execute(self, command: str):
        class DummyOutput:
            output: bytes
            exit_code: int

            def __init__(self, code, o):
                self.output = o
                self.exit_code = code

        if not isinstance(command, str):
            return DummyOutput(-1, b'')

        self._send_command(command)
        output = b''
        while True:
            try:
                data = self.sock.recv(8)
                if not data:
                    break
                _, n = struct.unpack('>BxxxL', data)
                line = self.sock.recv(n)
                output += line
                if re.search(b"\x1b.+@.+[#|$] ", line):
                    break
            except (TimeoutError, socket.timeout):
                break
        return DummyOutput(0, output)

    def execute_independent(self, command, attacker_identity, *params):
        # print("=== EXECUTING INDEPENDENT ===\n", command)
        language, command = command
        # if params:
        #     print("== Parameters ==\n", params)
        if language == "bash":
            cmd = ["bash", "-c", command]
            if params:
                cmd.append("--")
                cmd.extend(params)
        elif language == "python":
            cmd = ["python3", "-c", command, *params]
        elif language == "c++":
            self.execute_independent(("bash", f"echo \"{json.dumps(command)}\" > /tmp/main.cpp && "
                                              f"g++ -o /tmp/a.out /tmp/main.cpp"), None)
            cmd = ["/tmp/a.out", *params]
        elif language == "c":
            self.execute_independent(("bash", f"echo \"{json.dumps(command)}\" > /tmp/main.cpp && "
                                              f"gcc -o /tmp/a.out /tmp/main.cpp"), None)
            cmd = ["/tmp/a.out", *params]
        else:
            raise ValueError("Unsupported language")
        return self.container.exec_run(cmd, user=attacker_identity)
# 创建容器实例，并在同一个 shell 会话中执行命令

