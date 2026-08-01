"""Turn a ``diagnostic_writeup/`` directory into a standalone git project.

Each field's write-up is meant to become its own Overleaf document, so the
directory needs to be self-contained: a ``Makefile`` that builds the PDF, a
``.gitignore`` that keeps LaTeX's droppings out of history, a ``README`` that
says what the thing is, and a git repository with the generated state
committed.

The commit is deliberately made only when something changed, so re-running
the generator on an unchanged field is a no-op rather than a stream of empty
commits.
"""

import os
import subprocess

MAKEFILE = """# Build the diagnostic write-up.
#
#   make          -> main.pdf
#   make clean    -> remove LaTeX intermediates
#
# latexmk resolves the cross-references and the longtable page breaks in one
# invocation; a bare pdflatex needs three passes to settle.

main.pdf: main.tex $(wildcard figures/*.pdf)
\tlatexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
\tlatexmk -C
\trm -f main.bbl main.run.xml

.PHONY: clean
"""

GITIGNORE = """*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.synctex.gz
*.toc
main.pdf
"""

README = """# {field}: diagnostic write-up

Astrometric, photometric and diffuse-background ("background") diagnostics for
the JWST field **{field}**, generated from the released data products in

    {basepath}

## What this is, and what it is not

- **`JWST-GC/data-qa`** checks *initial* data products, close to the
  telescope, to catch a bad exposure early.
- **The astrometry paper** (Overleaf project `6a521006b63a11a7e0d80fa0`) is a
  technique-development document: the iterative work of establishing how to
  measure a position in these fields at all.
- **This document** is neither. It measures the properties of the *finished*
  products for one field, comprehensively, so that a user of the catalogue can
  see what precision they are entitled to assume.

## Building

    make

## Regenerating

From a checkout of `jwst-gc-pipeline`:

    python scripts/analysis/make_diagnostic_writeup.py --field {field}

Figures land in `figures/`, the text in `main.tex`, and every number the text
quotes in `measurements.json` — the prose is written from that file, so the
two cannot drift apart.

Generated {stamp} against pipeline version `{version}`.
"""


def _git(args, cwd):
    """Run a git command in *cwd*, returning (returncode, stdout+stderr)."""
    proc = subprocess.run(['git'] + args, cwd=cwd, capture_output=True,
                          text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def scaffold(outdir, field, basepath, stamp, version):
    """Write the Makefile, .gitignore and README into *outdir*."""
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'Makefile'), 'w') as fh:
        fh.write(MAKEFILE)
    with open(os.path.join(outdir, '.gitignore'), 'w') as fh:
        fh.write(GITIGNORE)
    with open(os.path.join(outdir, 'README.md'), 'w') as fh:
        fh.write(README.format(field=field, basepath=basepath, stamp=stamp,
                               version=version))


def init_and_commit(outdir, message):
    """Initialise a repository in *outdir* if needed and commit any change.

    Returns a short human-readable status string.
    """
    if not os.path.isdir(os.path.join(outdir, '.git')):
        code, out = _git(['init', '-q', '-b', 'main'], outdir)
        if code != 0:
            return f'git init failed: {out}'
    _git(['add', '-A'], outdir)
    code, _out = _git(['diff', '--cached', '--quiet'], outdir)
    if code == 0:
        return 'no changes to commit'
    code, out = _git(['commit', '-q', '-m', message], outdir)
    if code != 0:
        return f'commit failed: {out}'
    _code, rev = _git(['rev-parse', '--short', 'HEAD'], outdir)
    return f'committed {rev}'
