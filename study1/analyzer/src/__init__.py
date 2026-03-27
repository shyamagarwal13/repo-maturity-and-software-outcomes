"""Analysis modules for Study 1.

Some shared support modules live in the sibling collector component.
This package extends its ``__path__`` so that ``from src.data_models import``
style imports continue to resolve when the packaged ``collector/`` and
``analyzer/`` directories are used together.
"""

import os

def _load_env():
    """Read .env from project root without requiring python-dotenv."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = os.path.join(project_root, '.env')
    if not os.path.isfile(env_file):
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key not in os.environ:
                    os.environ[key] = value

_load_env()

_collector_path = os.environ.get('COLLECTOR_REPO_PATH', '../collector')
if _collector_path:
    # Resolve relative paths against project root
    if not os.path.isabs(_collector_path):
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _collector_path = os.path.join(_project_root, _collector_path)
    _collector_src = os.path.join(_collector_path, 'src')
    if os.path.isdir(_collector_src):
        __path__.append(_collector_src)
