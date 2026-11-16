"""EXR passes -> semantic / instance masks (numpy + OpenImageIO; runs in Blender's Python).

Masks come from the unfiltered Object Index pass: every pixel holds one
object's integer id, so no anti-aliasing can create invalid class values.
"""
import numpy as np
import OpenImageIO as oiio


def read_exr_layers(path):
    """{pass_name: (H, W, C) float32}; row 0 = top of the image."""
    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise IOError(oiio.geterror())
    out = {}
    sub = 0
    while inp.seek_subimage(sub, 0):
        spec = inp.spec()
        data = inp.read_image(sub, 0, 0, spec.nchannels, "float")
        names = list(spec.channelnames)
        groups = {}
        for k, n in enumerate(names):
            parts = n.split(".")
            key = ".".join(parts[:-1]) if len(parts) > 1 else n
            groups.setdefault(key, []).append(k)
        for key, idx in groups.items():
            short = key.split(".")[-1] if "." in key else key
            out[short] = np.asarray(data)[..., idx]
        sub += 1
    inp.close()
    return out


def write_png(path, arr, dtype):
    """Write an (h, w) or (h, w, c) array as PNG, row 0 = top.
    OpenImageIO's Python write_image scrambles a *2-D* array (verified with
    OIIO 3.1 in Blender 5.2 and 3.2 from pip), so single-channel images are
    written as (h, w, 1). Every file is read back and compared."""
    arr = arr.astype(dtype)
    if arr.ndim == 2:
        arr = arr[..., None]
    arr = np.ascontiguousarray(arr)
    h, w, c = arr.shape
    fmt = "uint8" if dtype == np.uint8 else "uint16"
    out = oiio.ImageOutput.create(path)
    out.open(path, oiio.ImageSpec(w, h, c, fmt))
    out.write_image(arr)
    out.close()
    back = oiio.ImageBuf(path).get_pixels(oiio.UINT8 if dtype == np.uint8 else oiio.UINT16)
    if back.shape != arr.shape or not np.array_equal(back, arr):
        raise IOError(f"PNG round-trip mismatch for {path}")


def make_masks(exr_path, label_table, classes, out_prefix):
    layers = read_exr_layers(exr_path)
    idx = np.rint(layers["Object Index"][..., 0]).astype(np.int64)
    depth = layers["Depth"][..., 0]
    max_id = max(label_table) if label_table else 0
    lut = np.full(max(max_id, int(idx.max())) + 1, classes["default_class"], np.int32)
    for iid, v in label_table.items():
        lut[iid] = v["class_id"]
    sem = lut[np.clip(idx, 0, len(lut) - 1)]
    sky = idx == 0
    sem[sky] = classes["sky_class"]
    unknown = (idx > max_id)
    if unknown.any():
        raise ValueError(f"{unknown.sum()} pixels carry ids not in the label table")
    valid_ids = {c["id"] for c in classes["classes"]}
    found = set(np.unique(sem).tolist())
    if not found <= valid_ids:
        raise ValueError(f"invalid class ids in mask: {found - valid_ids}")
    write_png(f"{out_prefix}_semantic.png", sem, np.uint8)
    pal = np.zeros((256, 3), np.uint8)
    for c in classes["classes"]:
        pal[c["id"]] = c["color"]
    write_png(f"{out_prefix}_semantic_color.png", pal[sem], np.uint8)
    shared = np.zeros(len(lut), bool)
    for iid, v in label_table.items():
        shared[iid] = v.get("shared", False)
    inst = np.where(sky | shared[np.clip(idx, 0, len(lut) - 1)], 0, idx)  # 'stuff' has no instances
    write_png(f"{out_prefix}_instance.png", inst, np.uint16)
    names = {c["id"]: c["name"] for c in classes["classes"]}
    hist = {names[k]: float((sem == k).mean()) for k in sorted(found)}
    inst_px = {}
    ids, counts = np.unique(idx[~sky], return_counts=True)
    for i, n in zip(ids.tolist(), counts.tolist()):
        inst_px[int(i)] = int(n)
    return {"class_fraction": hist, "instance_pixels": inst_px, "depth_range_m": [float(depth[~sky].min()), float(depth[~sky].max())] if (~sky).any() else None}
