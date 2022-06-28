from joblib import Memory
import definitions

disk_memory = Memory(definitions.ROOT_DIR / ".cache")
