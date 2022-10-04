from typing import Optional, Dict

class HFSpaceConfig:
    def __init__(
        self,
        app_path: str,
        repo_name: str,
        space_sdk: Optional[str] = "gradio",
        placeholders: Optional[Dict] = None
    ):
        self.app_path = app_path
        self.space_sdk = space_sdk
        self.placeholders = placeholders
        self.repo_name = repo_name