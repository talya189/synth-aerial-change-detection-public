"""JSON config loading with deep merge and project-relative paths."""
import copy
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Heavy generated data (extracted assets, texture tiers, DTM caches, renders,
# .blend files) lives outside the project: the project folder is inside an
# iCloud-synced Documents folder, where large writes are uploaded and reads can
# block while the file provider syncs. Override with SYNTH_WORK_ROOT.
WORK_ROOT = os.path.expanduser(os.environ.get("SYNTH_WORK_ROOT", "~/synth_aerial_work"))
WORK_DIRS = ("cache", "outputs", "templates", "data")


def project_path(*parts) -> str:
    p = os.path.join(*parts)
    if os.path.isabs(p):
        return p
    first = p.replace("\\", "/").split("/", 1)[0]
    return os.path.join(WORK_ROOT if first in WORK_DIRS else PROJECT_ROOT, p)


def rel_path(path) -> str:
    """Inverse of project_path: a path relative to the work root (for cache/,
    outputs/) or to the project root, so configs never hold absolute paths."""
    path = os.path.abspath(path)
    if path.startswith(WORK_ROOT + os.sep):
        return os.path.relpath(path, WORK_ROOT)
    return os.path.relpath(path, PROJECT_ROOT)


def load_json(path):
    with open(project_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path = project_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    raise TypeError(f"not JSON serialisable: {type(o)}")


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_scene_config(scene_config, overrides=None):
    """Scene config may name a parent preset via "extends"."""
    cfg = load_json(scene_config)
    if "extends" in cfg:
        parent = load_scene_config(os.path.join(os.path.dirname(scene_config), cfg.pop("extends")))
        cfg = deep_merge(parent, cfg)
    return deep_merge(cfg, overrides)
