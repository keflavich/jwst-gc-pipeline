"""The Aladin Lite control set every generated viewer shows.

Aladin Lite v3 hides most of its controls by default (3.8.2
``DEFAULT_OPTIONS``: the coordinate-grid, settings, colour-picker, share and
Simbad-pointer controls and the right-click menu are all ``false``).  The grid
colour, opacity and labels are set in the Settings menu (Settings > Grid), and
the grid-control button switches the grid on and off; with both hidden a reader
could neither show the grid nor change its magenta default, which is hard to
read against the treasury imagery.  Every viewer the pipeline writes now builds its options
from this one set, so a new control is enabled everywhere at once and no page
can fall behind the others.

The controls stay compact.  Each is a single toolbar icon that opens its panel on
click and closes it on a second click, and ``expandLayersControl`` is false so the
layer list also opens only when asked for.  A viewer with every control enabled
therefore still shows only a column of icons until the reader opens one.
"""

import json

# Every show* option Aladin Lite 3.8.2 accepts, all on.  Listed explicitly
# rather than relying on Aladin's defaults: several of those defaults are
# ``false``, and a future Aladin release could change the rest.
ALADIN_CONTROLS = {
    'showZoomControl': True,
    'showLayersControl': True,
    'expandLayersControl': False,
    'showFullscreenControl': True,
    'showSimbadPointerControl': True,
    'showCooGridControl': True,
    'showSettingsControl': True,
    'showColorPickerControl': True,
    'showShareControl': True,
    'showProjectionControl': True,
    'showFrame': True,
    'showFov': True,
    'showCooLocation': True,
    'showStatusBar': True,
    'showContextMenu': True,
}


def aladin_controls_js(**overrides):
    """Return the control options as JS object members, without braces.

    The result is spliced into an ``A.aladin(target, {...})`` literal next to
    the viewer's own options (survey, target, fov, ...), so it carries no
    surrounding ``{}`` -- the generators build their JS inside f-strings,
    where a bare brace would have to be escaped.

    ``overrides`` replace individual entries; an unknown key raises, so a
    misspelt control name fails at build time instead of being ignored by
    Aladin in the browser.
    """
    unknown = set(overrides) - set(ALADIN_CONTROLS)
    if unknown:
        raise KeyError(f'not an Aladin control option: {sorted(unknown)}')
    opts = {**ALADIN_CONTROLS, **overrides}
    return ', '.join(f'{k}: {json.dumps(v)}' for k, v in opts.items())
