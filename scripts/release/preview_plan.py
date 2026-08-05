"""Decide WHICH colour previews a staged field needs.

One preview per field is not enough for two kinds of field:

* **spatially independent pointings.**  gc2211 is four separate pointings of
  JWST 2211; a single image can only show one of them, and the other three are
  simply missing from the page.  Each pointing gets its own preview.
* **more filters than one RGB can carry.**  brick has ten NIRCam bands plus
  MIRI F2550W, sgrb2 fourteen in total.  A single R/G/B shows three of them and
  silently drops the rest, so every band a field ships must appear in at least
  one preview.

So the plan is: partition by pointing, then cover each pointing's filters with
consecutive-wavelength triples.  A field with two bands (gc2211) keeps the
existing two-channel R/(G=mean)/B and is only expanded spatially; a field with
fourteen gets five previews.

What counts as a separate pointing
----------------------------------
The staged layout is ``images/[<subdir>/]<FILTER>/``.  A subdir named like an
observation (``o023``) IS a distinct pointing.  ``MIRI`` is NOT -- it is the
same sky at a longer wavelength, so its filters join the field's pool and get
covered by the wavelength chunking like any other band.  That is what puts
brick's F2550W in an image instead of stranding it in a group of one.
"""
import os
import re

#: micron, for ordering previews by wavelength and labelling them.
FILTER_WAVELENGTH = {
    "F090W": 0.90, "F115W": 1.15, "F140M": 1.40, "F150W": 1.50, "F150W2": 1.50,
    "F162M": 1.62, "F182M": 1.82, "F187N": 1.87, "F200W": 2.00, "F210M": 2.10,
    "F212N": 2.12, "F277W": 2.77, "F300M": 3.00, "F322W2": 3.22, "F323N": 3.23,
    "F335M": 3.35, "F356W": 3.56, "F360M": 3.60, "F405N": 4.05, "F410M": 4.10,
    "F444W": 4.44, "F466N": 4.66, "F470N": 4.70, "F480M": 4.80,
    # MIRI
    "F560W": 5.6, "F770W": 7.7, "F1000W": 10.0, "F1130W": 11.3, "F1280W": 12.8,
    "F1500W": 15.0, "F1800W": 18.0, "F2100W": 21.0, "F2550W": 25.5,
}

#: A subdir that names a distinct pointing.  Anything else under ``images/``
#: (in practice ``MIRI``) is the same sky and is pooled with the field.
POINTING_RE = re.compile(r"^o\d{3}(-\d{3})?$")

FILTER_DIR_RE = re.compile(r"^F\d{3,4}[WMN]2?$")


def _filter_dirs(root):
    """``{FILTER: subdir}`` for one directory level, ``subdir`` '' at the root."""
    out = {}
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        if FILTER_DIR_RE.match(name) and os.path.isdir(os.path.join(root, name)):
            out[name] = ""
    return out


def staged_groups(field_dir):
    """``{pointing: {FILTER: subdir}}``.

    ``pointing`` is ``None`` for the field itself.  ``subdir`` is what
    ``make_preview_rgb`` has to search to find that filter ('' = the images
    root, 'MIRI' = the MIRI subdir), which is why the value is not just a set.
    """
    images = os.path.join(str(field_dir), "images")
    groups = {None: _filter_dirs(images)}
    if not os.path.isdir(images):
        return {}
    for name in sorted(os.listdir(images)):
        path = os.path.join(images, name)
        if not os.path.isdir(path) or FILTER_DIR_RE.match(name):
            continue
        found = _filter_dirs(path)
        if not found:
            continue
        if POINTING_RE.match(name):
            groups.setdefault(name, {}).update({f: "" for f in found})
        else:
            # same sky, longer wavelength (MIRI) -> pool with the field
            groups[None].update({f: name for f in found})
    return {k: v for k, v in groups.items() if v}


def chunk_filters(filters, size=3):
    """Consecutive-wavelength chunks covering every filter exactly once.

    A trailing chunk of one cannot make an image, so it borrows from the chunk
    before it -- 7 filters become 3+2+2, never 3+3+1.
    """
    ordered = sorted(filters, key=lambda f: (FILTER_WAVELENGTH.get(f, 99.0), f))
    if len(ordered) <= size:
        return [ordered] if ordered else []
    chunks = [ordered[i:i + size] for i in range(0, len(ordered), size)]
    if len(chunks[-1]) == 1:
        tail = chunks.pop()
        chunks.append([chunks[-1].pop()] + tail)
    return chunks


def plan(field_dir):
    """Every preview a staged field should have.

    Each entry is ``{pointing, filters, subdirs}``: ``filters`` is R..B ordered
    longest-to-shortest wavelength (red = reddest, which is what an RGB should
    do), and ``subdirs`` are the extra ``images/`` subdirectories that have to
    be searched to find them.
    """
    out = []
    groups = sorted(staged_groups(field_dir).items(),
                    key=lambda kv: (kv[0] is not None, kv[0] or ""))
    for pointing, found in groups:
        extra = sorted({sub for sub in found.values() if sub})
        for chunk in chunk_filters(list(found)):
            out.append({
                "pointing": pointing,
                # reddest first: R, [G], B
                "filters": sorted(chunk, key=lambda f: -FILTER_WAVELENGTH.get(f, 99.0)),
                "subdirs": extra,
            })
    return out


def describe(spec):
    """Human label for a preview, e.g. 'o023 - F277W/F200W'."""
    bands = "/".join(spec["filters"])
    return f"{spec['pointing']} - {bands}" if spec["pointing"] else bands
