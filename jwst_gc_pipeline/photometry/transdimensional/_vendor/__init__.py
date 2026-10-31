"""Code vendored from PCAT-DE (https://github.com/RichardFeder/pcat-de, commit 033aa43).

Copyright (c) 2022 Richard Feder, MIT License (``LICENSE_PCAT-DE.txt`` in this
directory).  PCAT-DE descends from the Lion probabilistic cataloger of
Portillo, Lee, Schlafly & Finkbeiner (2017, AJ 154, 132) and Feder et al.
(2020, AJ 159, 163).

* ``pcat_lion_kernel.c``  model + likelihood C kernel (patched, see header).
* ``image_eval.py``       ``psf_poly_fit`` and ``image_model_eval`` reference
  implementation (patched, see docstring); used by the tests as the numpy
  reference for the C kernel.

The sampler in ``..pcat_sampler`` is a new implementation of the PCAT
proposal scheme; it does not import PCAT-DE's ``pcat_main.py`` (Python 3 but
not importable: hard ``matplotlib.use('TkAgg')``, missing ``pcat_blas_routines``
module, ``np.int``/``np.bool`` removed in numpy >= 1.24).
"""
