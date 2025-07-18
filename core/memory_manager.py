from __future__ import annotations

"""Gestor de memoria de usuario (simple)
Se guarda un diccionario {user_id: UserMemory} en un archivo pickle.
El user_id puede ser por ahora 'default'.
"""

import pickle
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

MEMORY_FILE = Path(__file__).parent.parent / "data" / "users_memory.pkl"
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

@dataclass
class UserMemory:
    last_patient_id: Optional[str] = None
    last_tables: List[str] = field(default_factory=list)
    preferences: Dict[str, str] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)  # Últimas consultas (texto)
    updated_at: datetime = field(default_factory=datetime.utcnow)


def load_memory() -> Dict[str, UserMemory]:
    if MEMORY_FILE.exists():
        try:
            with MEMORY_FILE.open("rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def save_memory(mem: Dict[str, UserMemory]):
    try:
        with MEMORY_FILE.open("wb") as f:
            pickle.dump(mem, f)
    except Exception:
        pass 