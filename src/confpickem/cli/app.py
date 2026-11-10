#!/usr/bin/env python
"""Launch the live tracker web app (Streamlit). Needs: pip install "confpickem[app]"."""

import sys
from pathlib import Path


def main():
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print('❌ Streamlit not installed. Run: pip install -e ".[app]"')
        return 1
    script = Path(__file__).parent.parent / "live_app.py"
    sys.argv = ["streamlit", "run", str(script), *sys.argv[1:]]
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())
