# ...existing code...
import sys
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).resolve().parents[0] / ".env"
if not env_path.exists():
    env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))

try:
    from pipeline.umls_loader import UMLSLoader
except Exception as e:
    print(f"✗ Failed to import UMLSLoader: {e}")
    sys.exit(1)

def normalize_cui(s: str) -> str:
    s = s.strip()
    if s.upper().startswith("UMLS:"):
        s = s.split(":", 1)[1]
    return s.upper()

def _read_api_key_fallback() -> str:
    """Try env then manual .env parse as fallback."""
    key = os.getenv("UMLS_API_KEY", "") or ""
    if key:
        return key.strip()
    # fallback parse .env near repo root
    try:
        p = Path(__file__).resolve().parents[1] / ".env"
        if p.exists():
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip().startswith("UMLS_API_KEY"):
                    parts = ln.split("=", 1)
                    if len(parts) == 2:
                        return parts[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""

def main():
    cui = sys.argv[1] if len(sys.argv) > 1 else "C5958835"
    cui = normalize_cui(cui)

    api_key = _read_api_key_fallback()
    if not api_key:
        print("✗ UMLS_API_KEY not found. Add UMLS_API_KEY to .env or environment.")
        sys.exit(2)

    loader = UMLSLoader(api_key=api_key)

    # prefer is_available() if implemented
    try:
        available = loader.is_available() if hasattr(loader, "is_available") else True
    except Exception:
        available = False

    if not available:
        print("✗ UMLS not available. Check network or API key.")
        sys.exit(2)

    # try common method names, fallback gracefully
    details = None
    try:
        details = loader.get_concept_details(cui)
    except AttributeError:
        try:
            details = loader.get_concept(cui)
        except Exception:
            details = None
    except Exception as e:
        print(f"✗ Error querying UMLS: {e}")
        sys.exit(3)

    if not details:
        print(f"ℹ No details found for UMLS:{cui}")
        sys.exit(0)

    print(json.dumps({"UMLS": cui, "details": details}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
# ...existing code...