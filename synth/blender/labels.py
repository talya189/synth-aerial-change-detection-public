"""Instance id registry and class mapping.

Every labelled object gets `pass_index = instance_id` (exact in float32 EXR
up to 2^24). The Cycles Object Index pass is unfiltered, so each pixel holds
exactly one object's id. Classes are resolved from each instance's `kind`
through configs/classes.json, so changing the taxonomy never touches generators.
"""
import fnmatch


PASS_INDEX_MAX = 32767  # Object.pass_index hard limit (verified in Blender 5.2)


class IdRegistry:
    """Countable kinds (classes.json `instance_kinds`) get one id per object;
    other kinds share one id per kind. Raises instead of letting Blender clamp."""

    def __init__(self, classes):
        self.items = {}
        self._next = 1
        self._shared = {}
        self._by_plan = {}
        self._patterns = classes.get("instance_kinds", ["*"])

    def new(self, kind, plan_id):
        """Instance id for an entity. The same (kind, plan_id) always gets the same
        id, so all parts of one entity (a grove's trees, a road's turning circle)
        form one instance."""
        if not any(fnmatch.fnmatch(kind, p) for p in self._patterns):
            if kind not in self._shared:
                self._shared[kind] = self._alloc(kind, f"<all {kind}>", shared=True)
            return self._shared[kind]
        key = (kind, plan_id)
        if key not in self._by_plan:
            self._by_plan[key] = self._alloc(kind, plan_id, shared=False)
        return self._by_plan[key]

    def _alloc(self, kind, plan_id, shared):
        iid = self._next
        if iid > PASS_INDEX_MAX:
            raise OverflowError(f"more than {PASS_INDEX_MAX} labelled instances; add kinds to shared 'stuff'")
        self._next += 1
        self.items[iid] = {"kind": kind, "plan_id": plan_id, "shared": shared}
        return iid


def class_of(kind, classes):
    for c in classes["classes"]:
        for pat in c.get("kinds", []):
            if fnmatch.fnmatch(kind, pat):
                return c["id"]
    return classes["default_class"]


def label_table(ids, classes):
    return {iid: {**v, "class_id": class_of(v["kind"], classes)} for iid, v in ids.items.items()}
