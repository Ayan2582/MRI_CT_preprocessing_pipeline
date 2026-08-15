"""
config.py
─────────
YAML configuration with inheritance and dotted command-line overrides.

The whole point of this model's design is that the loss is reconfigurable
without editing code:

    python model/scripts/train.py --config model/configs/exp2_paper.yaml \
        --set loss.lambda_nce=0 loss.lambda_l1=100

turns the paper loss into plain pix2pix. So the config layer has three jobs:

  1. `_base_` inheritance, so the five experiment files hold only their deltas
     and every shared default lives in exactly one place (base.yaml).
  2. Dotted `--set a.b.c=value` overrides that reject unknown keys. A silent
     typo like `loss.lamda_nce=0` would otherwise produce a run that looks like
     the experiment you asked for and is not, which is unrecoverable after the
     fact because nothing in the logs would disagree.
  3. A stable hash of the fully-resolved config, stored in every checkpoint, so
     `--resume` can tell you when you are about to continue a run with different
     hyper-parameters than it started with.
"""

import copy
import hashlib
import json
import logging
import os

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


# ── Dotted access ─────────────────────────────────────────────────────────────

class Config(dict):
    """
    A dict that also supports attribute and dotted-path access.

    `cfg.loss.lambda_nce` and `cfg["loss.lambda_nce"]` both work, which keeps
    call sites readable without forcing every consumer to remember which nesting
    level a knob lives at.
    """

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(
                f"No config key '{key}'. Available at this level: "
                f"{sorted(self.keys())}"
            ) from None
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key, value):
        self[key] = value

    def get_path(self, dotted, default=None):
        """Return the value at 'a.b.c', or `default` if any level is missing."""
        node = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


# ── Loading ───────────────────────────────────────────────────────────────────

def _deep_merge(base, override):
    """
    Recursively merge `override` into `base`, returning a new dict.

    Dicts merge key-by-key; every other type (including lists) replaces
    wholesale. Lists replace rather than concatenate because the one list that
    matters here is `loss.nce.layers` — merging [0,2,4] into [0,2,4,6,8] to get
    a seven-element tap list would be nonsense.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml_with_base(path, _seen=None):
    """Load a YAML file, resolving its `_base_` chain depth-first."""
    path = os.path.abspath(path)
    _seen = _seen or []
    if path in _seen:
        chain = " -> ".join(os.path.basename(p) for p in _seen + [path])
        raise ValueError(f"Circular _base_ reference in configs: {chain}")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    base_ref = raw.pop("_base_", None)
    if base_ref is None:
        return raw

    # A _base_ is resolved relative to the referring file, so configs/ can be
    # copied elsewhere (e.g. onto Kaggle) as a unit and still resolve.
    base_path = os.path.join(os.path.dirname(path), base_ref)
    base = _load_yaml_with_base(base_path, _seen + [path])
    return _deep_merge(base, raw)


# ── Overrides ─────────────────────────────────────────────────────────────────

def _coerce(text):
    """
    Turn a command-line string into a typed value.

    YAML's scalar parser is reused so that `0`, `0.0`, `true`, `null` and
    `[0,2,4]` all coerce the way they would if written in the config file
    itself. A bare string that YAML cannot parse comes back unchanged.
    """
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(cfg, overrides):
    """
    Apply a list of 'dotted.key=value' strings in place.

    Every path must already exist in the config. This is the strictness that
    makes a typo'd experiment impossible: base.yaml is the schema, and anything
    not declared there is a mistake rather than a new setting.
    """
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(
                f"Malformed override '{item}'. Expected the form key.subkey=value, "
                f"for example loss.lambda_nce=0"
            )
        dotted, _, raw_value = item.partition("=")
        dotted = dotted.strip()
        parts = dotted.split(".")

        node = cfg
        for i, part in enumerate(parts[:-1]):
            if part not in node or not isinstance(node[part], dict):
                so_far = ".".join(parts[:i + 1])
                raise KeyError(
                    f"Override '{dotted}' does not match the config: "
                    f"'{so_far}' is not a section. "
                    f"Available: {sorted(node.keys()) if isinstance(node, dict) else node}"
                )
            node = node[part]

        leaf = parts[-1]
        if leaf not in node:
            raise KeyError(
                f"Override '{dotted}' sets an unknown key '{leaf}'. "
                f"Keys at that level: {sorted(node.keys())}. "
                f"(Every valid knob is declared in configs/base.yaml.)"
            )

        old = node[leaf]
        node[leaf] = _coerce(raw_value)
        logger.info("override  %s: %r -> %r", dotted, old, node[leaf])

    return cfg


# ── Hashing ───────────────────────────────────────────────────────────────────

# Keys that describe *where* a run happens rather than *what* it computes.
# They are excluded from the hash so that moving a run from this machine to
# Kaggle — different data root, different device, different worker count —
# does not read as "you changed the model" on resume.
_HASH_EXCLUDE = {
    "data.root",
    "data.manifest",
    "data.splits",
    "run.out_dir",
    "run.name",
    "runtime.device",
    "runtime.num_workers",
    "runtime.amp",
    "runtime.pin_memory",
    "logging.sample_every",
    "logging.print_every",
}


def config_hash(cfg):
    """
    Return a short stable hash of the semantically meaningful config.

    Used to warn on `--resume` when the checkpoint was produced by a different
    configuration. It is a warning and not an error: legitimately extending
    `train.n_epochs` on a second Kaggle session must not be blocked.
    """
    def strip(node, prefix=""):
        out = {}
        for key in sorted(node):
            dotted = f"{prefix}{key}"
            if dotted in _HASH_EXCLUDE:
                continue
            # Underscore-prefixed keys are bookkeeping this module added itself
            # (_config_path, _hash, _resolved_from). dump_config writes them
            # into config.resolved.yaml, so including them here would make the
            # hash of a reloaded config differ from the hash it was saved with —
            # and every evaluate.py run would report a spurious CONFIG MISMATCH.
            if key.startswith("_"):
                continue
            value = node[key]
            out[key] = strip(value, dotted + ".") if isinstance(value, dict) else value
        return out

    payload = json.dumps(strip(dict(cfg)), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ── Entry point ───────────────────────────────────────────────────────────────

def load_config(path, overrides=None):
    """
    Resolve a config file plus CLI overrides into a Config.

    `path` may be a filename inside model/configs (e.g. 'exp2_paper.yaml') or
    any path; the bare-name form is what the Kaggle notebook uses.
    """
    if not os.path.isfile(path):
        candidate = os.path.join(CONFIG_DIR, path)
        if os.path.isfile(candidate):
            path = candidate

    cfg = Config(_load_yaml_with_base(path))
    apply_overrides(cfg, overrides)
    cfg["_config_path"] = os.path.abspath(path)
    cfg["_hash"] = config_hash(cfg)
    return cfg


def dump_config(cfg, path):
    """Write the fully-resolved config beside a run's checkpoints."""
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    payload["_resolved_from"] = cfg.get("_config_path")
    payload["_hash"] = cfg.get("_hash")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False)
    os.replace(tmp, path)      # atomic, matching the convention used repo-wide
