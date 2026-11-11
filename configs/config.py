"""Read YAML configuration and expand the workflow's data paths."""
import argparse
from pathlib import Path
import re

import yaml


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CODE_ROOT / 'configs' / 'step1.yaml'
REFERENCE = re.compile(r'\$\{([a-zA-Z_][a-zA-Z0-9_.]*)\}')
# Existing path fields predating the _dir / _path naming convention.
LEGACY_PATH_KEYS = {'alfworld_games', 'train_latents', 'eval_latents'}


def read_yaml(path):
    with Path(path).open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def load_config(path=DEFAULT_CONFIG):
    config = read_yaml(path)
    def resolve(value, key='', trail=()):
        if isinstance(value, dict):
            return {k: resolve(v, k, trail) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(item, key, trail) for item in value]
        if not isinstance(value, str):
            return value

        def reference(match):
            name = match[1]
            if name in trail:
                raise ValueError(f"Circular config reference: {' -> '.join((*trail, name))}")
            referenced = config
            keys = name.split('.')
            for part in keys:
                referenced = referenced[part]
            return resolve(referenced, keys[-1], (*trail, name))

        def replace(match):
            referenced = reference(match)
            if isinstance(referenced, (dict, list)):
                raise TypeError(f'Cannot embed config container in a string: {match[1]}')
            return str(referenced)

        match = REFERENCE.fullmatch(value)
        value = reference(match) if match else REFERENCE.sub(replace, value)
        # Hugging Face's name_or_path may be a repository ID, not a local path.
        is_path = isinstance(key, str) and key != 'name_or_path' and (
            key.endswith(('_dir', '_path')) or key in LEGACY_PATH_KEYS
        )
        if isinstance(value, str) and is_path:
            value = str((CODE_ROOT / Path(value).expanduser()).resolve())
        return value

    return resolve(config)


def config_for_parser(parser, argv=None):
    parser.add_argument('--config', default=str(DEFAULT_CONFIG), help='experiment YAML')
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--config', default=str(DEFAULT_CONFIG))
    args, _ = preliminary.parse_known_args(argv)
    return load_config(args.config)
