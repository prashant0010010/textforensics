"""Entry point for a PyInstaller build (see README, 'Standalone executable'). Untested in the build environment."""
import os
import sys


def main() -> None:
    from streamlit.web import cli as stcli
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.argv = ["streamlit", "run", os.path.join(base, "app.py"), "--global.developmentMode=false",
                "--browser.gatherUsageStats=false"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
