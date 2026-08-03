"""Monitoring for jwst-gc-pipeline runs.

Answers, from the products on disk plus the SLURM queue, how far a run has got,
whether what it produced is coherent, and what is in flight -- for a full field
or for a small ``--cutout-region`` probe run.

    python -m jwst_gc_pipeline.monitoring                  # write the pages
    python -m jwst_gc_pipeline.monitoring --target brick   # one field
    python -m jwst_gc_pipeline.monitoring probe --execute  # submit 5" probes

Modules
-------
``scan``    on-disk product taxonomy, observation-safe.
``jobs``    squeue/sacct state and log error signatures.
``checks``  verdicts, with every threshold imported from its enforcing module.
``probe``   plan and submit the tiny cutout runs the monitor is exercised on.
``render``  self-contained HTML.
``report``  glue + file output.
"""
from .report import build_entries, write_report, summarize   # noqa: F401

__all__ = ['build_entries', 'write_report', 'summarize']
