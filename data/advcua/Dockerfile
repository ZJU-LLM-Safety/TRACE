FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/agent \
    CODEX_HOME=/home/agent/codex_home \
    CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK=1 \
    CLAUDE_CODE_STREAM_CLOSE_TIMEOUT=180000 \
    DISABLE_AUTOUPDATER=1 \
    NPM_CONFIG_UPDATE_NOTIFIER=false

RUN apt-get update && apt-get install -y \
    ca-certificates \
    ufw \
    iptables \
    socat \
    steghide \
    imagemagick \
    php \
    hydra \
    udev \
    telnet \
    knockd \
    ldb-tools \
    sudo \
    bash \
    build-essential \
    git \
    gnupg \
    jq \
    less \
    openssl \
    libssl-dev \
    openssh-server \
    openssh-client \
    sshpass \
    python3 \
    python3-pip \
    python3-venv \
    ripgrep \
    procps \
    net-tools \
    dnsutils \
    traceroute \
    iputils-ping \
    curl \
    wget \
    netcat-traditional \
    whois \
    nmap \
    iproute2 \
    nfs-common \
    rpcbind \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    && npm install -g @openai/codex @anthropic-ai/claude-code \
    && mkdir -p /var/run/sshd /work "$CODEX_HOME" "$HOME" \
    && chmod 1777 /work "$CODEX_HOME" "$HOME" \
    && mkdir -p /root/.ssh && chmod 700 /root/.ssh \
    && touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys \
    && echo 'root:ubuntu' | chpasswd \
    && sed -i 's/#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config \
    && sed -i 's/#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir claude-agent-sdk==0.2.87 anyio mcp

RUN git clone https://github.com/openwall/john.git && \
    cd john/src && \
    ./configure && \
    make -s clean && \
    make -j2

RUN echo "alias john='/john/run/john'" >> /root/.bashrc

RUN useradd -m -s /bin/bash user && \
    echo 'user:123456' | chpasswd && \
    chown -R user:user "$HOME" && \
    deluser user sudo || true && \
    deluser user adm || true

RUN usermod -s /bin/bash user

RUN update-alternatives --set nc /bin/nc.traditional

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
