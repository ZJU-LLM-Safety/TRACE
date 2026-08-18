FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/agent
ENV CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK=1
ENV CLAUDE_CODE_STREAM_CLOSE_TIMEOUT=180000

ARG INSTALL_CODEX=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    gnupg \
    jq \
    nodejs \
    npm \
    python3 \
    python3-pip \
    ripgrep \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --timeout 120 --retries 5 \
    claude-agent-sdk==0.2.87 \
    anyio \
    mcp \
    openai

# Optional Codex CLI support for --backend codex. The normal AgentHazard path
# uses Claude Code SDK, so this is off by default to keep builds simpler.
RUN if [ "$INSTALL_CODEX" = "true" ]; then \
      npm install -g @openai/codex; \
    fi

RUN useradd -m -s /bin/bash agent \
 && mkdir -p /work /harness /home/agent/.claude /home/agent/.config /home/agent/.npm \
 && chown -R agent:agent /work /harness /home/agent

USER agent
WORKDIR /work

# The run script overrides the entrypoint and bind-mounts a per-run Python
# driver into /bridge. This default keeps the image inspectable by hand.
ENTRYPOINT ["python3", "-c", "print('ar_agenthazard image ready; use tools/agenthazard_bridge_run.py')"]
