#!/bin/bash
# 停掉 DFlash2 vLLM。
# launcher 是用 setsid/nohup 把 api_server 甩到独立会话里的，systemd 管不到它的 cgroup，
# 所以只能按进程名杀。放在脚本文件里执行，命令行本身不含这些模式，不会自杀。
for p in 'VLLM::Engin[e]Core' 'VLLM::Worke[r]' \
         'vllm-2080ti-0.2[.]x/.venv/bin/python' \
         'entrypoints.openai.api_serve[r]' \
         'launcher[.]sh'; do
  pkill -9 -f "$p" 2>/dev/null
done
sleep 5
exit 0
