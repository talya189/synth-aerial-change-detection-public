"""Asset ingest (standard Python venv).

    python -m synth.prep.asset_ingest [--tiers 1024 2048] [--force]

1. Extracts every zip in the asset root with Python's zipfile (macOS `unzip`
   cannot read several Megascans archives) into cache/assets/<stem>/.
2. Reads the Megascans <id>.json for type and physical size.
3. Classifies texture maps by filename suffix and writes downscaled tiers to
   cache/tex/<stem>/<tier>/ (Cycles memory: 4K maps are ~64 MB each).
4. Writes configs/assets.json. Hand-edited fields of existing entries are kept.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import zipfile

from PIL import Image

from synth.common.config import load_json, project_path, rel_path, save_json

Image.MAX_IMAGE_PIXELS = None

MAP_PATTERNS = [
    ("base_color", r"(basecolor|albedo|diffuse)"),
    ("roughness", r"roughness"),
    ("gloss", r"gloss"),
    ("normal", r"normal"),
    ("height", r"(displacement|height)"),
    ("ao", r"(_ao|ambientocclusion)"),
    ("opacity", r"opacity"),
    ("translucency", r"translucency"),
    ("metallic", r"(metalness|metallic)"),
]
NON_COLOR = {"roughness", "gloss", "normal", "height", "ao", "opacity", "metallic"}
MODEL_EXT = (".fbx", ".glb", ".gltf", ".obj", ".usd", ".usda", ".usdc", ".blend")


def classify_map(fname):
    low = os.path.basename(fname).lower()
    if not low.endswith((".jpg", ".jpeg", ".png", ".exr", ".tif", ".tiff")):
        return None
    if "billboard" in low:
        return None  # billboard cards are useless for 3D views
    for key, pat in MAP_PATTERNS:
        if re.search(pat, low):
            return key
    return None


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def extract(src, dst):
    """Returns (ok, message)."""
    if os.path.isdir(src):
        return True, "folder"
    try:
        with zipfile.ZipFile(src) as z:
            done_flag = os.path.join(dst, ".extracted")
            if os.path.exists(done_flag):
                return True, "cached"
            os.makedirs(dst, exist_ok=True)
            z.extractall(dst)
            open(done_flag, "w").close()
            return True, "extracted"
    except zipfile.BadZipFile as e:
        return False, f"corrupt zip ({os.path.getsize(src)} bytes): {e}"


def read_megascans_json(folder):
    for f in glob.glob(os.path.join(folder, "**", "*.json"), recursive=True):
        if "DynamicWind" in f:
            continue
        try:
            j = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if "meta" not in j and "semanticTags" not in j:
            continue
        meta = {m.get("key"): m.get("value") for m in j.get("meta", []) if isinstance(m, dict)}
        info = {"megascans_id": j.get("id"), "kind": j.get("semanticTags", {}).get("asset_type"),
                "categories": j.get("categories", []), "average_color": j.get("averageColor")}
        if "scanArea" in meta:
            w, h = re.findall(r"[\d.]+", meta["scanArea"])[:2]
            info["real_size_m"] = [float(w), float(h)]
            if meta.get("height"):
                info["height_m"] = float(re.findall(r"[\d.]+", meta["height"])[0])
        dims = [meta.get(k) for k in ("length", "width", "height")]
        if all(dims) and info.get("kind") in ("3D asset",):
            info["dims_m"] = [float(re.findall(r"[\d.]+", d)[0]) for d in dims]
        if "tileable" in meta:
            info["tileable"] = bool(meta["tileable"])
        return info
    return {}


def make_tiers(stem, maps, tiers, force=False):
    out = {}
    for key, src in maps.items():
        out[key] = {}
        with Image.open(src) as im:
            full = max(im.size)
            for t in tiers:
                if t >= full:
                    out[key][str(t)] = rel_path(src)
                    continue
                dst = project_path("cache", "tex", stem, str(t), os.path.basename(src))
                if force or not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    scale = t / full
                    size = (max(1, round(im.size[0] * scale)), max(1, round(im.size[1] * scale)))
                    img = im
                    if img.mode not in ("RGB", "L", "RGBA"):
                        img = img.convert("RGB")
                    img.resize(size, Image.LANCZOS).save(dst, quality=93)
                out[key][str(t)] = rel_path(dst)
        out[key]["full"] = rel_path(src)
    return out


def ingest(tiers=(1024, 2048), force=False):
    roles_cfg = load_json("configs/asset_roles.json")
    asset_root = os.path.normpath(project_path(roles_cfg["asset_root"]))
    registry_path = project_path("configs/assets.json")
    registry = load_json("configs/assets.json") if os.path.exists(registry_path) else {"assets": {}}
    report = {"ok": [], "corrupt": [], "unassigned": []}

    sources = {}
    for p in sorted(os.listdir(asset_root)):
        full = os.path.join(asset_root, p)
        if p.startswith("."):
            continue
        stem = p[:-4] if p.endswith(".zip") else p
        if os.path.isdir(full) and os.path.exists(full + ".zip"):
            continue  # prefer the zip; the folder is a partial manual extract
        if p.endswith(".zip") or os.path.isdir(full):
            sources[stem] = full

    for stem, src in sources.items():
        dst = project_path("cache", "assets", stem)
        ok, msg = extract(src, dst) if src.endswith(".zip") else (True, "folder")
        folder = dst if src.endswith(".zip") else src
        if not ok:
            report["corrupt"].append(f"{os.path.basename(src)}: {msg}")
            registry["assets"].setdefault(stem, {})["status"] = "corrupt"
            continue
        files = [f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True) if os.path.isfile(f)]
        maps = {}
        for f in sorted(files):
            k = classify_map(f)
            # prefer jpg over exr, and the highest resolution
            if k and (k not in maps or (maps[k].lower().endswith(".exr") and not f.lower().endswith(".exr"))):
                maps[k] = f
        models = sorted(rel_path(f) for f in files if f.lower().endswith(MODEL_EXT))
        info = read_megascans_json(folder)
        entry = registry["assets"].get(stem, {})
        auto = {
            "status": "ok",
            "source": rel_path(src),
            "path": rel_path(folder),
            "models": models,
            "maps": make_tiers(stem, maps, tiers, force),
            "non_color_maps": sorted(NON_COLOR & set(maps)),
            "normal_convention": "opengl",
            "version": f"megascans:{info.get('megascans_id')}",
            "source_size_bytes": os.path.getsize(src) if os.path.isfile(src) else None,
            **info,
        }
        # auto-detected fields refresh; hand-edited extra fields survive
        entry.update(auto)
        entry["roles"] = roles_cfg["roles"].get(stem, entry.get("roles", []))
        if not entry["roles"]:
            report["unassigned"].append(stem)
        registry["assets"][stem] = entry
        report["ok"].append(f"{stem}: {info.get('kind')} maps={sorted(maps)} models={len(models)} ({msg})")

    save_json(registry, "configs/assets.json")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    rep = ingest(tuple(a.tiers), a.force)
    for line in rep["ok"]:
        print("OK      ", line)
    for line in rep["corrupt"]:
        print("CORRUPT ", line)
    for line in rep["unassigned"]:
        print("NO ROLE ", line)
    print(f"{len(rep['ok'])} ok, {len(rep['corrupt'])} corrupt, {len(rep['unassigned'])} without role")
    return 1 if rep["corrupt"] else 0


if __name__ == "__main__":
    sys.exit(main())
