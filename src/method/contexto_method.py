"""
AMA-Bench method adapter for Contexto.

Bridges to the @ekai/mindmap package via a lightweight HTTP server.
Implements BaseMethod with memory_construction (add) and memory_retrieve (search).
"""

import os
import requests
from typing import Any, Dict, List, Optional

from src.method.base_method import BaseMethod


class ContextoMethod(BaseMethod):
    def __init__(
        self,
        config_path: Optional[str] = None,
        bridge_url: Optional[str] = None,
    ):
        self.config: Dict[str, Any] = {}
        if config_path:
            self.config = self._load_config(config_path)

        self.bridge_url = (
            bridge_url
            or self.config.get("bridge_url")
            or os.environ.get("CONTEXTO_BRIDGE_URL")
            or "http://localhost:3456"
        )

        self._episode_counter = 0

        # Verify bridge is reachable
        try:
            resp = requests.get(f"{self.bridge_url}/health", timeout=5)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Contexto bridge server not reachable at {self.bridge_url}. "
                "Start it with: bun src/server.ts"
            )

    def memory_construction(self, traj_text: str, task: str = "") -> Dict[str, Any]:
        """Parse trajectory text and add to mindmap via bridge server."""
        self._episode_counter += 1
        episode_id = str(self._episode_counter)

        items = self._parse_trajectory(traj_text, task, episode_id)

        resp = requests.post(
            f"{self.bridge_url}/construct",
            json={
                "episodeId": episode_id,
                "items": items,
            },
            timeout=300,
        )
        resp.raise_for_status()
        result = resp.json()

        return {"episode_id": episode_id, "total_items": result.get("totalItems", 0)}

    def memory_retrieve(self, memory: Any, question: str) -> str:
        """Retrieve relevant context from mindmap via bridge server."""
        episode_id = memory["episode_id"]

        resp = requests.post(
            f"{self.bridge_url}/retrieve",
            json={
                "episodeId": episode_id,
                "question": question,
            },
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()

        return result.get("context", "")

    def _parse_trajectory(
        self, traj_text: str, task: str, episode_id: str
    ) -> List[Dict[str, Any]]:
        """
        Parse trajectory text into ConversationItem format.
        Splits on 'Turn N' / 'Step N' markers (matching AMA-Bench's own splitting).
        Falls back to 500-char chunks if no markers found.
        """
        full_text = traj_text
        if task:
            full_text = f"Task: {task}\n\n{traj_text}"

        # Split on Turn/Step markers
        lines = full_text.split("\n")
        segments: List[str] = []
        current_segment: List[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Turn ") or stripped.startswith("Step "):
                if current_segment:
                    segments.append("\n".join(current_segment))
                    current_segment = []
            current_segment.append(line)

        if current_segment:
            segments.append("\n".join(current_segment))

        # Fallback: chunk into 500-char segments
        if not segments:
            chunk_size = 500
            segments = [
                full_text[i : i + chunk_size]
                for i in range(0, len(full_text), chunk_size)
            ]

        # Convert to ConversationItem format
        items = []
        for idx, segment in enumerate(segments):
            content = segment.strip()
            if not content:
                continue

            # Infer role from content
            role = "user"
            if "Action:" in content:
                role = "assistant"
            elif "Observation:" in content:
                role = "system"

            items.append({
                "id": f"ep{episode_id}_turn{idx}",
                "role": role,
                "content": content,
                "metadata": {"episodeId": episode_id, "turnIndex": idx},
            })

        return items
