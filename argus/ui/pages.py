"""Import this module to register every ``@ui.page`` route as a side effect."""

import argus.ui.dashboard as dashboard
import argus.ui.picks as picks
import argus.ui.settings as settings
import argus.ui.sources as sources

__all__ = ["dashboard", "picks", "settings", "sources"]
