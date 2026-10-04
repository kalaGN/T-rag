from pathlib import Path
import yaml
from t_rag.web.app import create_web_app


def test_real_gradio_builds_three_tabs(tmp_path):
    settings = tmp_path / "settings.yaml"
    settings.write_text(yaml.safe_dump({"data_dir": str(tmp_path / "data")}))
    app = create_web_app(settings)
    config = app.get_config_file()
    tabs = [component["props"]["label"] for component in config["components"] if component["type"] == "tabitem"]
    assert tabs == ["知识库", "文档与重建", "问答"]
    assert not any("业务域" in str(component["props"]) for component in config["components"])


def test_initial_library_and_status_are_consistent(tmp_path):
    from t_rag.config import Settings
    from t_rag.knowledge.service import KnowledgeService
    directory = tmp_path / "data"
    item = KnowledgeService(Settings(data_dir=str(directory))).create("已有知识库")
    settings = tmp_path / "settings.yaml"
    settings.write_text(yaml.safe_dump({"data_dir": str(directory)}))
    components = create_web_app(settings).get_config_file()["components"]
    selector = next(c for c in components if c["type"] == "dropdown" and c["props"]["label"] == "当前知识库")
    assert selector["props"]["value"] == item["id"]
    assert any(c["type"] == "markdown" and "未建索引" in c["props"]["value"] for c in components)
