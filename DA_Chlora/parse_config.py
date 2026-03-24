import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime

import yaml

from logger import setup_logging


# ---------------------------------------------------------------------------
# YAML I/O helpers (replaces the previous read_json / write_json dependency)
# ---------------------------------------------------------------------------

def read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_yaml(content, path):
    with open(path, "w") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# ConfigParser
# ---------------------------------------------------------------------------

class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None, uid=None, weights_dir=None, is_training=True):
        """
        Parse a YAML configuration file and set up the experiment environment.

        Parameters
        ----------
        config : dict
            Contents of the config.yml file (already loaded).
        resume : str | None
            Path to a checkpoint to resume from.
        modification : dict | None
            {keychain: value} pairs to override config values at runtime.
            Key chains use ';' as separator, e.g. 'optimizer;args;lr'.
        run_id : str | None
            Explicit run identifier.  Defaults to a timestamp + hyper-param suffix.
        weights_dir : str | Path | None
            Override for tester.weights_dir (CLI takes precedence over config).
        """
        self._config = _update_config(config, modification)
        self.resume = resume

        # ------------------------------------------------------------------
        # Validate / auto-fill arch.in_channels from data_loader.input_vars
        # ------------------------------------------------------------------
        self._sync_in_channels_with_data_loader()

        # ------------------------------------------------------------------
        # Sync arch.dataset_type from data_loader so model and data match
        # ------------------------------------------------------------------
        self._sync_dataset_type_with_data_loader()

        # ------------------------------------------------------------------
        # Build directory paths
        # ------------------------------------------------------------------
        save_dir = Path(self._config["trainer"]["save_dir"])
        exper_name = self._config["name"]

        # Build run_id from uid prefix + config-derived suffix, or auto-generate.
        # Both train.py and test.py apply the same formula, so passing the same
        # -i/--uid to both scripts always resolves to the same directory.
        prev_days  = self._config["data_loader"]["args"]["previous_days"]
        activation = self._config["arch"]["args"]["hidden_activation"]
        suffix = f"prevdays_{prev_days}_activation_{activation}"

        if uid is not None:
            run_id = uid + suffix
        elif run_id is None:
            run_id = datetime.now().strftime(r"%m%d_%H%M%S") + suffix
        # else: explicit run_id passed (e.g. resume), use as-is

        self._save_dir = save_dir / "models" / exper_name / run_id
        self._log_dir  = save_dir / "logs"   / exper_name / run_id

        if is_training:
            # Create the run directory, persist config, write last_run.txt.
            self._weights_dir = None
            exist_ok = run_id == ""
            self._save_dir.mkdir(parents=True, exist_ok=exist_ok)
            self._log_dir.mkdir(parents=True, exist_ok=exist_ok)

            write_yaml(self._config, self._save_dir / "config.yml")

            breadcrumb = save_dir / "last_run.txt"
            breadcrumb.write_text(str(self._save_dir))
        else:
            # Testing: uid → reconstruct same path; else fall back to -wd / config.
            if uid is not None:
                self._weights_dir = self._save_dir
            else:
                self._weights_dir = self._resolve_weights_dir(weights_dir)
            self._log_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        setup_logging(self._log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_weights_dir(self, cli_weights_dir):
        """
        Determine the effective weights_dir for test mode.

        Priority (highest to lowest):
          1. CLI argument (--weights_dir)
          2. tester.weights_dir in config (absolute path)
          3. tester.weights_dir in config treated as a run_id relative to
             trainer.save_dir/models/<name>/
          4. last_run.txt breadcrumb written by the most recent training run
        """
        # 1. CLI override
        if cli_weights_dir is not None:
            return Path(cli_weights_dir)

        cfg_wd = self._config.get("tester", {}).get("weights_dir", None)

        # 2 & 3. From config
        if cfg_wd is not None:
            p = Path(cfg_wd)
            if p.is_absolute():
                return p
            # Treat as run_id relative to the canonical models directory
            save_dir   = Path(self._config["trainer"]["save_dir"])
            exper_name = self._config["name"]
            return save_dir / "models" / exper_name / p

        # 4. Breadcrumb from last training run
        save_dir    = Path(self._config["trainer"]["save_dir"])
        breadcrumb  = save_dir / "last_run.txt"
        if breadcrumb.exists():
            resolved = Path(breadcrumb.read_text().strip())
            logging.getLogger(__name__).info(
                f"weights_dir not specified; using last training run from {breadcrumb}: {resolved}"
            )
            return resolved

        # Nothing found - train.py is fine with None; test.py raises its own clear error.
        return None

    def _infer_day_channels_from_data_loader(self):
        dl      = self._config.get("data_loader", {})
        dl_args = dl.get("args", {}) if isinstance(dl, dict) else {}
        if not isinstance(dl_args, dict):
            return None

        # Preferred: explicitly named variables
        input_vars = dl_args.get("input_vars", None)
        if input_vars is not None:
            if not isinstance(input_vars, list) or not all(isinstance(v, str) for v in input_vars):
                raise ValueError(
                    f"`data_loader.args.input_vars` must be a list of strings, got: {input_vars}"
                )
            if len(input_vars) == 0:
                raise ValueError("`data_loader.args.input_vars` cannot be empty.")
            return len(input_vars)

        # Backwards-compatible: integer indices or name strings
        selected_vars = dl_args.get("selected_vars", None)
        if selected_vars is not None:
            if not isinstance(selected_vars, list):
                raise ValueError(
                    f"`data_loader.args.selected_vars` must be a list, got: {selected_vars}"
                )
            if len(selected_vars) == 0:
                raise ValueError("`data_loader.args.selected_vars` cannot be empty.")
            if not (
                all(isinstance(v, int) for v in selected_vars)
                or all(isinstance(v, str) for v in selected_vars)
            ):
                raise ValueError(
                    "`data_loader.args.selected_vars` must be all ints or all strings."
                )
            return len(selected_vars)

        return None

    def _sync_in_channels_with_data_loader(self):
        arch      = self._config.get("arch", {})
        arch_args = arch.get("args", {}) if isinstance(arch, dict) else {}
        if not isinstance(arch_args, dict):
            return

        inferred = self._infer_day_channels_from_data_loader()
        if inferred is None:
            return

        current = arch_args.get("in_channels", None)
        if current is None or current == "auto":
            arch_args["in_channels"] = inferred
            return

        if not isinstance(current, int):
            raise ValueError(
                f"`arch.args.in_channels` must be an int (or null/'auto'), "
                f"got: {current} ({type(current)})"
            )

        if current != inferred:
            raise ValueError(
                "Config mismatch: `arch.args.in_channels` must match the number of "
                "input variables per day.\n"
                f"  arch.args.in_channels          = {current}\n"
                f"  inferred from data_loader       = {inferred}\n"
                "Fix: set `arch.args.in_channels` to null (auto) or align "
                "`data_loader.args.input_vars`."
            )

    def _sync_dataset_type_with_data_loader(self):
        """Set arch.args.dataset_type from data_loader.args.dataset_type so model and data stay in sync."""
        dl = self._config.get("data_loader", {})
        dl_args = dl.get("args", {}) if isinstance(dl, dict) else {}
        if not isinstance(dl_args, dict):
            return

        dl_dataset_type = dl_args.get("dataset_type", None)
        if dl_dataset_type is None:
            return

        arch = self._config.get("arch", {})
        arch_args = arch.get("args", {}) if isinstance(arch, dict) else {}
        if not isinstance(arch_args, dict):
            return

        current = arch_args.get("dataset_type", None)
        if current is None or current == "auto":
            arch_args["dataset_type"] = dl_dataset_type
            return

        if current != dl_dataset_type:
            raise ValueError(
                "Config mismatch: `arch.args.dataset_type` must match `data_loader.args.dataset_type`.\n"
                f"  arch.args.dataset_type         = {current!r}\n"
                f"  data_loader.args.dataset_type  = {dl_dataset_type!r}\n"
                "Fix: set `arch.args.dataset_type` to null (auto) or align with data_loader."
            )

    # ------------------------------------------------------------------
    # Class method – initialise from CLI args
    # ------------------------------------------------------------------

    @classmethod
    def from_args(cls, args, options=()):
        """
        Initialise ConfigParser from CLI arguments.

        Recognised standard flags
        -------------------------
        -c / --config       Path to config.yml
        -r / --resume       Path to a checkpoint
        -d / --device       CUDA_VISIBLE_DEVICES string
        --weights_dir       Override tester.weights_dir (test.py only)
        """
        known, unknown = args.parse_known_args()
        if unknown:
            print(f"Warning: unknown arguments will be ignored: {unknown}")

        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)

        if not isinstance(args, tuple):
            known = args.parse_known_args()[0]

        if known.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = known.device

        if known.resume is not None:
            resume   = Path(known.resume)
            cfg_path = resume.parent / "config.yml"
        else:
            assert known.config is not None, (
                "A configuration file must be specified.  "
                "Add '-c config.yml', for example."
            )
            resume   = None
            cfg_path = Path(known.config)

        config = read_yaml(cfg_path)
        if known.config and resume:
            # Allow a fresh config to override a resumed checkpoint's config
            config.update(read_yaml(known.config))

        modification = {
            opt.target: getattr(known, _get_opt_name(opt.flags))
            for opt in options
        }

        # -i/--uid is registered by both scripts; -wd only by test.py.
        # The absence of "weights_dir" on the namespace is the signal we are
        # in training mode (test.py is the only one that registers --weights_dir).
        is_training = not hasattr(known, "weights_dir")
        uid         = getattr(known, "uid", None)
        weights_dir = getattr(known, "weights_dir", None)

        return cls(config, resume, modification, uid=uid, weights_dir=weights_dir, is_training=is_training)

    # ------------------------------------------------------------------
    # Module initialisation helpers
    # ------------------------------------------------------------------

    def init_obj(self, name, module, *args, **kwargs):
        """
        `config.init_obj('key', module)` ≡ `module.ClassName(*args, **cfg_args, **kwargs)`
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert not any(k in module_args for k in kwargs), (
            "Overwriting kwargs given in config file is not allowed."
        )
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        `config.init_ftn('key', module)` ≡ `partial(module.ClassName, *args, **cfg_args)`
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert not any(k in module_args for k in kwargs), (
            "Overwriting kwargs given in config file is not allowed."
        )
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    # ------------------------------------------------------------------
    # Dict-like access
    # ------------------------------------------------------------------

    def __getitem__(self, name):
        return self._config[name]

    def get_logger(self, name, verbosity=2):
        assert verbosity in self.log_levels, (
            f"verbosity option {verbosity} is invalid. "
            f"Valid options are {list(self.log_levels)}."
        )
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        """Training checkpoint directory (populated during training)."""
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def weights_dir(self):
        """Resolved weights directory for test mode."""
        return self._weights_dir


# ---------------------------------------------------------------------------
# Config mutation helpers
# ---------------------------------------------------------------------------

def _update_config(config, modification):
    if modification is None:
        return config
    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith("--"):
            return flg.lstrip("-").replace("-", "_")
    return flags[0].lstrip("-").replace("-", "_")


def _set_by_path(tree, keys, value):
    """Set a value in a nested dict by a ';'-separated key chain."""
    keys = keys.split(";")
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Retrieve a value from a nested dict by a sequence of keys."""
    return reduce(getitem, keys, tree)
