import json
import os
import subprocess
import sys


def test_cli_create_list_and_required_knowledge_base(tmp_path):
    env = {**os.environ, "PYTHONPATH": "src", "T_RAG_DATA_DIR": str(tmp_path / "data")}
    command = [sys.executable, "-m", "t_rag.cli"]
    created = subprocess.run([*command, "kb", "create", "CLI test"], env=env, capture_output=True, text=True)
    assert created.returncode == 0, created.stderr
    kb_id = json.loads(created.stdout)["id"]
    listed = subprocess.run([*command, "kb", "list"], env=env, capture_output=True, text=True)
    assert listed.returncode == 0
    assert json.loads(listed.stdout)[0]["id"] == kb_id
    missing = subprocess.run([*command, "query", "question"], env=env, capture_output=True, text=True)
    assert missing.returncode != 0
    assert "--kb" in missing.stderr
