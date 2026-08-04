"""The Aladin Lite sky view: where the survey plans to observe, and what it has.

Modelled on ``roman_footprint_gbtds.html`` from the galactic-plane-surveys repo,
with the layer set inverted for this page's purpose: **only the JWST footprints
are on by default**, and the Roman GBTDS tiles are available but off, because a
pipeline monitor is about the JWST survey and the Roman geometry is context.

Three JWST layers, deliberately separate:

* **planned NIRCam** -- the prime instrument's 8 SW detectors per pointing;
* **planned MIRI** -- the coordinated parallel, which lands ~7.5' away from the
  prime and so covers *different sky*.  Drawing it in the NIRCam colour would
  suggest the survey covers a contiguous area it does not;
* **observed** -- its own colour and its own toggle, filled in from the APT
  visit status.  Today it is empty; the layer and its toggle still render, so
  the page shows "nothing observed yet" rather than hiding the question.

Loading
-------
Aladin Lite is a ~1.8 MB script and the sky view is not what most visits to this
page are for, so it loads **lazily** on first expand.  The script is served from
this page's own directory (``report.publish`` links a copy in), so the page
depends on no third-party CDN.

The HiPS image tiles are unavoidably remote -- that is what a HiPS is.  Where
those requests are blocked (a published artifact runs under a strict CSP, and a
page opened from ``file://`` has no origin), the panel says so and points at the
served copy rather than showing an empty black box.
"""
import json
import os

#: Where a copy of Aladin Lite is hardlinked from when publishing.  Same origin
#: as the page, so no third-party CDN is involved.
ALADIN_SOURCE = '/orange/adamginsburg/web/public/ACES_Aladin_tour/aladin.js'
ALADIN_LOCAL = 'aladin.js'

#: Written next to the page by ``report.write_report``.
FOOTPRINTS_JSON = 'footprints.json'

#: Layer colours.  JWST uses the page's own accent family; Roman keeps the
#: blue/orange of the source page so the two are recognisably the same layers.
COLOR_NIRCAM_PLANNED = '#46bcd6'
COLOR_MIRI_PLANNED = '#a78bfa'
COLOR_OBSERVED = '#4ade80'
COLOR_SPRING = '#1E90FF'
COLOR_AUTUMN = '#FF8C00'
COLOR_TARGET_AREA = '#ff6b6b'

#: Backgrounds worth having here.  The survey's own imagery first.
SURVEYS = (
    ('JWST CMZ', 'https://starformation.astro.ufl.edu/avm_images/jwst_cmz_hips/'),
    ('DSS', 'P/DSS2/color'),
    ('2MASS', 'P/2MASS/color'),
    ('WISE', 'P/allWISE/color'),
    ('GLIMPSE', 'P/Spitzer/GLIMPSE360'),
)


def load_footprints(path):
    """The footprint JSON, or ``None``.  Built by ``scripts/monitoring/build_footprints.py``."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _esc(text):
    import html
    return html.escape('' if text is None else str(text), quote=True)


CSS = """
.gcm-sky { margin-top: .75rem; }
.gcm-sky-wrap { position: relative; height: 560px; border-radius: 3px;
                overflow: hidden; border: 1px solid var(--rule);
                background: #05080a; }
#gcm-aladin { position: absolute; inset: 0; }
.gcm-sky-ui { position: absolute; top: 8px; right: 8px; z-index: 10; width: 208px;
              background: rgba(10,16,20,.86); color: #dfe9ec;
              border: 1px solid rgba(255,255,255,.13); border-radius: 4px;
              font-size: 11.5px; overflow: hidden;
              font-family: var(--sans); backdrop-filter: blur(6px); }
.gcm-sky-ui h4 { margin: 0; padding: 6px 10px; font-size: 11px; font-weight: 600;
                 letter-spacing: .06em; text-transform: uppercase;
                 background: rgba(255,255,255,.06); color: #9fb4bc;
                 border-bottom: 1px solid rgba(255,255,255,.1);
                 font-family: var(--mono); }
.gcm-sky-sec { padding: 6px 9px; border-bottom: 1px solid rgba(255,255,255,.07); }
.gcm-sky-sec:last-child { border-bottom: 0; }
.gcm-sky-lab { font-size: 9.5px; text-transform: uppercase; letter-spacing: .07em;
               color: #7d919a; margin-bottom: 4px; font-family: var(--mono); }
.gcm-sky-row { display: flex; gap: 4px; flex-wrap: wrap; }
.gcm-sky-btn { cursor: pointer; padding: 3px 7px; border-radius: 3px;
               border: 1px solid rgba(255,255,255,.18);
               background: rgba(255,255,255,.06); color: #cfdde2;
               font-size: 10.5px; font-family: var(--mono); }
.gcm-sky-btn:hover { background: rgba(255,255,255,.16); color: #fff; }
.gcm-sky-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.gcm-sky-btn.on { border-color: currentColor; font-weight: 600; }
.gcm-sky-count { color: #6b8089; font-size: 9.5px; margin-left: .25rem; }
.gcm-sky-empty { color: #8b6f3a; }
.gcm-sky-msg { position: absolute; inset: 0; display: flex; align-items: center;
               justify-content: center; text-align: center; padding: 2rem;
               color: #9fb4bc; font-size: .85rem; line-height: 1.6; z-index: 5; }
.gcm-sky-msg a { color: var(--accent); }
.gcm-sky-load { display: inline-block; cursor: pointer; margin-top: .6rem;
                padding: .35rem .8rem; border-radius: 3px;
                border: 1px solid var(--accent); color: var(--accent);
                background: none; font-family: var(--mono); font-size: .78rem; }
.gcm-sky-legend { display: grid; grid-template-columns: 15px 1fr; gap: 3px 6px;
                  align-items: center; }
.gcm-sky-sw { width: 15px; height: 3px; border-radius: 2px; }
"""


def section(footprints, roman=None, aladin_src=ALADIN_LOCAL,
            data_url=FOOTPRINTS_JSON):
    """The whole sky-view section, or a note when there is no footprint data."""
    if not footprints:
        return ('<section class="gcm-sec" id="skyview"><h2>Sky view</h2>'
                '<p class="gcm-empty">No footprint data — run '
                '<code>scripts/monitoring/build_footprints.py</code> to generate '
                '<code>footprints.json</code>.</p></section>')

    n_planned = footprints.get('n_planned', 0)
    n_observed = footprints.get('n_observed', 0)
    lo, hi = (footprints.get('pa_v3_range') or [None, None])[:2]
    pa_note = (f"PA_V3 {footprints.get('pa_v3', 0):.0f}°"
               + (f" (program allows {lo:.0f}–{hi:.0f}°)"
                  if lo is not None and hi is not None else ''))

    observed_cls = 'gcm-sky-empty' if not n_observed else ''
    roman_json = json.dumps(roman or {})

    surveys = ''.join(
        f'<button class="gcm-sky-btn survey{" on" if i == 0 else ""}" '
        f'data-survey="{_esc(url)}">{_esc(name)}</button>'
        for i, (name, url) in enumerate(SURVEYS))

    return f"""
<section class="gcm-sec gcm-sky" id="skyview"><h2>Sky view — survey footprints</h2>
<p class="gcm-note">Program {_esc(footprints.get('program'))},
<em>{_esc(footprints.get('title'))}</em>: {n_planned} planned pointings, NIRCam
prime with MIRI as a coordinated parallel. The MIRI parallel sits ~7.5′ from the
prime, so it covers <em>different sky</em> — it is a separate layer for that
reason, not for tidiness. {_esc(pa_note)}; the footprints rotate within that
range until each visit is scheduled, so treat the exact corners as indicative.
Observed pointings come from the APT visit status: <strong>{n_observed}</strong>
so far.</p>

<div class="gcm-sky-wrap">
  <div id="gcm-aladin"></div>
  <div class="gcm-sky-msg" id="gcm-sky-msg">
    <div>Interactive sky view — Aladin Lite plus remote HiPS imagery.
      <br><button class="gcm-sky-load" id="gcm-sky-load" type="button">load sky view</button>
    </div>
  </div>

  <div class="gcm-sky-ui" id="gcm-sky-ui" hidden>
    <h4>Footprints</h4>

    <div class="gcm-sky-sec">
      <div class="gcm-sky-lab">JWST — planned</div>
      <div class="gcm-sky-row">
        <button class="gcm-sky-btn on" id="lyr-nircam"
                style="color:{COLOR_NIRCAM_PLANNED}">NIRCam
          <span class="gcm-sky-count">{n_planned}</span></button>
        <button class="gcm-sky-btn on" id="lyr-miri"
                style="color:{COLOR_MIRI_PLANNED}">MIRI ∥
          <span class="gcm-sky-count">{n_planned}</span></button>
      </div>
    </div>

    <div class="gcm-sky-sec">
      <div class="gcm-sky-lab">JWST — observed</div>
      <div class="gcm-sky-row">
        <button class="gcm-sky-btn on" id="lyr-observed"
                style="color:{COLOR_OBSERVED}">observed
          <span class="gcm-sky-count {observed_cls}">{n_observed or 'none yet'}</span></button>
      </div>
    </div>

    <div class="gcm-sky-sec">
      <div class="gcm-sky-lab">Roman GBTDS (context)</div>
      <div class="gcm-sky-row">
        <button class="gcm-sky-btn" id="lyr-spring"
                style="color:{COLOR_SPRING}">spring</button>
        <button class="gcm-sky-btn" id="lyr-autumn"
                style="color:{COLOR_AUTUMN}">autumn</button>
        <button class="gcm-sky-btn" id="lyr-target"
                style="color:{COLOR_TARGET_AREA}">target area</button>
      </div>
    </div>

    <div class="gcm-sky-sec">
      <div class="gcm-sky-lab">Background</div>
      <div class="gcm-sky-row">{surveys}</div>
    </div>

    <div class="gcm-sky-sec">
      <div class="gcm-sky-lab">Legend</div>
      <div class="gcm-sky-legend">
        <div class="gcm-sky-sw" style="background:{COLOR_NIRCAM_PLANNED}"></div>
        <div>NIRCam planned</div>
        <div class="gcm-sky-sw" style="background:{COLOR_MIRI_PLANNED}"></div>
        <div>MIRI parallel planned</div>
        <div class="gcm-sky-sw" style="background:{COLOR_OBSERVED}"></div>
        <div>observed</div>
      </div>
    </div>
  </div>
</div>
</section>

<script>
(function () {{
  var DATA_URL = {json.dumps(data_url)};
  var ALADIN_SRC = {json.dumps(aladin_src)};
  var ROMAN = {roman_json};
  var C = {json.dumps({'nircam': COLOR_NIRCAM_PLANNED, 'miri': COLOR_MIRI_PLANNED,
                       'observed': COLOR_OBSERVED, 'spring': COLOR_SPRING,
                       'autumn': COLOR_AUTUMN, 'target': COLOR_TARGET_AREA})};

  var btn = document.getElementById('gcm-sky-load');
  var msg = document.getElementById('gcm-sky-msg');
  var ui = document.getElementById('gcm-sky-ui');
  if (!btn) {{ return; }}

  function fail(what) {{
    // Say which part is unavailable and where it does work, rather than
    // leaving a black rectangle that looks like a broken page.
    msg.innerHTML = '<div>Sky view unavailable here — ' + what +
      '.<br><span style="font-size:.8em;opacity:.8">Aladin needs its script and '
      'remote HiPS tiles; a published artifact blocks both, and a page opened '
      'from a file:// path has no origin to fetch from. It works on the served '
      'copy.</span></div>';
    msg.hidden = false;
  }}

  btn.onclick = function () {{
    btn.disabled = true;
    btn.textContent = 'loading…';
    var s = document.createElement('script');
    s.src = ALADIN_SRC;
    s.onerror = function () {{ fail('the Aladin script could not be loaded'); }};
    s.onload = function () {{
      fetch(DATA_URL).then(function (r) {{
        if (!r.ok) {{ throw new Error('http ' + r.status); }}
        return r.json();
      }}).then(start).catch(function (e) {{
        fail('the footprint data could not be loaded (' + e.message + ')');
      }});
    }};
    document.body.appendChild(s);
  }};

  function start(fp) {{
    A.init.then(function () {{
      var aladin = A.aladin('#gcm-aladin', {{
        survey: {json.dumps(SURVEYS[0][1])},
        target: '0 0', fov: 1.6, cooFrame: 'galactic'
      }});
      msg.hidden = true;
      ui.hidden = false;

      function layer(name, color, width) {{
        var ov = A.graphicOverlay({{ color: color, lineWidth: width || 1.2,
                                     name: name }});
        aladin.addOverlay(ov);
        return ov;
      }}

      var L = {{
        nircam: layer('JWST NIRCam (planned)', C.nircam, 1.1),
        miri: layer('JWST MIRI parallel (planned)', C.miri, 1.1),
        observed: layer('JWST observed', C.observed, 2.0),
        spring: layer('Roman GBTDS spring', C.spring, 1.2),
        autumn: layer('Roman GBTDS autumn', C.autumn, 1.2),
        target: layer('JWST target area', C.target, 2.0)
      }};

      (fp.planned || []).forEach(function (p) {{
        (p.nircam || []).forEach(function (poly) {{ L.nircam.add(A.polygon(poly)); }});
        (p.miri || []).forEach(function (poly) {{ L.miri.add(A.polygon(poly)); }});
      }});
      (fp.observed || []).forEach(function (p) {{
        (p.nircam || []).forEach(function (poly) {{ L.observed.add(A.polygon(poly)); }});
        (p.miri || []).forEach(function (poly) {{ L.observed.add(A.polygon(poly)); }});
      }});
      Object.keys(ROMAN.tiles || {{}}).forEach(function (name) {{
        var t = ROMAN.tiles[name];
        (t.spring || []).forEach(function (poly) {{ L.spring.add(A.polygon(poly)); }});
        (t.autumn || []).forEach(function (poly) {{ L.autumn.add(A.polygon(poly)); }});
      }});
      if (ROMAN.target_area) {{ L.target.add(A.polygon(ROMAN.target_area)); }}

      // Default state: the JWST layers only. Everything else exists but is
      // off -- this is a JWST pipeline monitor, and the Roman geometry is
      // context. `observed` is ON despite being empty today, so the first
      // executed visit appears without anyone remembering to enable it.
      var on = {{ nircam: true, miri: true, observed: true,
                 spring: false, autumn: false, target: false }};
      function apply() {{
        Object.keys(L).forEach(function (k) {{
          on[k] ? L[k].show() : L[k].hide();
        }});
      }}
      apply();

      [['lyr-nircam', 'nircam'], ['lyr-miri', 'miri'],
       ['lyr-observed', 'observed'], ['lyr-spring', 'spring'],
       ['lyr-autumn', 'autumn'], ['lyr-target', 'target']
      ].forEach(function (pair) {{
        var el = document.getElementById(pair[0]);
        if (!el) {{ return; }}
        el.classList.toggle('on', on[pair[1]]);
        el.onclick = function () {{
          on[pair[1]] = !on[pair[1]];
          el.classList.toggle('on', on[pair[1]]);
          apply();
        }};
      }});

      document.querySelectorAll('#gcm-sky-ui button.survey').forEach(function (b) {{
        b.onclick = function () {{
          document.querySelectorAll('#gcm-sky-ui button.survey')
            .forEach(function (o) {{ o.classList.remove('on'); }});
          b.classList.add('on');
          var t = b.dataset.survey;
          aladin.setImageSurvey(t.indexOf('http') === 0 ? A.HiPS(t) : t);
        }};
      }});
    }});
  }}
}})();
</script>"""
