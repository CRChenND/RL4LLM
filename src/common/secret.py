import os, configparser, pathlib

def load_secret_from_ini(path: str, key: str) -> str:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing secrets file: {path}")
    cfg = configparser.ConfigParser()
    cfg.read(path)
    if "secrets" not in cfg or key not in cfg["secrets"]:
        raise KeyError(f"Key '{key}' not found under [secrets] in {path}")
    return cfg["secrets"][key]

def ensure_env_from_ini(path: str, env_var: str, key: str):
    if os.environ.get(env_var):  # already set
        return os.environ[env_var]
    val = load_secret_from_ini(path, key)
    os.environ[env_var] = val
    return val
