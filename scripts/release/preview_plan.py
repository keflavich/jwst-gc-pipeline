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

#: ``F164N`` is 1.64 um, ``F2550W`` is 25.50: the numeric code IS the wavelength
#: in units of 0.01 um, for every filter this pipeline stages (checked against
#: the 35-entry table this replaces -- all agreed to <0.02 um).
#:
#: It is DERIVED rather than looked up because a hand-maintained table defaulted
#: a missing filter to 99 um, which sorted it into the RED channel with a bluer
#: band in blue -- wd1 shipped "R=F164N ... B=F466N" (1.64 um in red, 4.66 in
#: blue) and wd2 "R=F164N ... B=F250M", on live pages, under captions asserting
#: the ordering.  A guard test against the field registry did not catch it,
#: because those filters reach a page through the staged tree and are not in the
#: registry at all.  Derivation removes the failure mode instead of watching for it.
_FILTER_CODE_RE = re.compile(r"^F(\d{3,4})[WMN]2?$")


def wavelength_um(filt):
    """Pivot wavelength in micron, from the filter name.  Raises on a name this
    does not understand -- silently guessing is what put a 1.64 um band in the
    red channel."""
    match = _FILTER_CODE_RE.match(str(filt).upper())
    if match is None:
        raise ValueError(f"cannot derive a wavelength from filter name {filt!r}")
    return int(match.group(1)) / 100.0


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
        path = os.path.join(root, name)
        if not (FILTER_DIR_RE.match(name) and os.path.isdir(path)):
            continue
        # `isdir` alone is not "this filter is staged": a band whose mosaics were
        # quarantined (`*_i2d_im0_badastrom.fits`) leaves the directory behind,
        # and it would enter the plan and then die at render time with
        # FileNotFoundError -- after earlier previews had already been written.
        if not any(f.endswith("_i2d.fits") for f in os.listdir(path)):
            continue
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
    ordered = sorted(filters, key=lambda f: (wavelength_um(f), f))
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
        if len(found) < 2:
            # One band cannot make an RGB.  Emitting the spec anyway reached
            # `--filters takes 2 or 3 filter names` inside the recursive main()
            # and exited 2 -- taking the whole run with it, mid-way, leaving a
            # partial gallery under an "every band appears" caption.
            print(f"  preview plan: skipping {pointing or 'field'} -- only "
                  f"{sorted(found)} staged, which cannot make a colour image")
            continue
        for chunk in chunk_filters(list(found)):
            out.append({
                "pointing": pointing,
                # reddest first: R, [G], B
                "filters": sorted(chunk, key=lambda f: -wavelength_um(f)),
                "subdirs": extra,
            })
    return out


def describe(spec):
    """Human label for a preview, e.g. 'o023 - F277W/F200W'."""
    bands = "/".join(spec["filters"])
    return f"{spec['pointing']} - {bands}" if spec["pointing"] else bands
