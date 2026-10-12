import os

# All defaults below are read from environment variables so this package
# carries no machine-specific paths. Set these in your shell (or a .env
# loaded before import) to match your own storage layout, or simply pass
# --base / --scan-dirs / --output-dir explicitly on every CLI call instead.
DEFAULT_BASE = os.environ.get("LLMTRAITS_BASE", "")
SCAN_DIRS = [p for p in os.environ.get("LLMTRAITS_SCAN_DIRS", "").split(os.pathsep) if p]
DEFAULT_OUTPUT = os.environ.get("LLMTRAITS_DELTAS_DIR", "./deltas")
DEFAULT_DTYPE = "float16"
KNOWN_DUPLICATES: set[str] = set()

LLAMA_HIDDEN = 8192
LLAMA_INTERMEDIATE = 28672
LLAMA_VOCAB_MAIN = 128256

LAYER_SHARD_PREFIX = "shard_layer"
GLOBAL_SHARD_NAME = "shard_global.safetensors"
MANIFEST_NAME = "manifest.json"
