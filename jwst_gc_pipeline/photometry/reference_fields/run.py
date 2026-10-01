"""Run the reference fields through the cutout pipeline (one SLURM job per run).

::

    # print the jobs for the code in this checkout, labelled 'main'
    python -m jwst_gc_pipeline.photometry.reference_fields.run --variant main

    # submit them, pinned to a worktree
    python -m jwst_gc_pipeline.photometry.reference_fields.run --variant qfit \\
        --pipe-root /path/to/wt-qfit --submit

Each field runs once clean (seed 0) and once per injection seed, every filter
of the field in one process (the m7 cross-band seed needs all of them).  A
run writes ``<basepath>/cutouts/ref_<field>_<variant>_s<seed>/``; an existing
run directory is removed first unless ``--keep``, because the cutout pipeline
resumes from whatever it finds there and a variant must not inherit another
code's intermediate products.
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys

from jwst_gc_pipeline.photometry import reference_fields as RF

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
PYTHON = '/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python'
LOG_DIR = '/orange/adamginsburg/jwst/logs/reference_fields'


def pipeline_command(spec, variant, seed, python=PYTHON, extra_args=()):
    """The ``crowdsource_catalogs_long`` command of one run."""
    region = f"{spec['ra']:.7f},{spec['dec']:.7f},{spec['size_arcsec']:g}"
    cmd = [python, '-m', 'jwst_gc_pipeline.photometry.crowdsource_catalogs_long',
           f"--proposal_id={spec['proposal']}", f"--field={spec['obsid']}",
           f"--target={spec['target']}", f"--filternames={','.join(spec['filters'])}",
           f"--modules={spec.get('modules', 'merged')}", '--each-exposure',
           f"--each-suffix={spec['each_suffix']}",
           f'--cutout-region={region}',
           f'--cutout-label={RF.run_label(spec["name"], variant, seed)}']
    if spec.get('each_suffix_overrides'):
        cmd.append(f"--each-suffix-overrides={spec['each_suffix_overrides']}")
    if int(seed):
        cmd += [f'--inject-stars={RF.injection_table_path(spec["name"], seed)}',
                f'--inject-seed={int(seed)}']
    cmd += list(spec.get('extra_args', [])) + list(extra_args)
    return cmd


def sbatch_command(spec, variant, seed, pipe_root, *, cpus, mem, walltime,
                   python=PYTHON, extra_args=(), env=()):
    """``sbatch`` argv of one run (job name says field, variant and seed)."""
    label = RF.run_label(spec['name'], variant, seed)
    inner = pipeline_command(spec, variant, seed, python=python,
                             extra_args=list(extra_args) + [f'--parallel-workers={cpus}'])
    env = list((spec.get('env') or {}).items()) + list(env)
    exports = ' '.join(f'export {k}={shlex.quote(str(v))};' for k, v in env)
    wrap = (f'export PYTHONPATH={shlex.quote(pipe_root)}:${{PYTHONPATH:-}}; {exports} '
            f'cd {shlex.quote(pipe_root)}; ' + shlex.join(inner))
    return ['sbatch', '--parsable', f'--job-name=reffield-{label}',
            '--account=astronomy-dept', '--qos=astronomy-dept-b',
            f'--cpus-per-task={cpus}', f'--mem={mem}', f'--time={walltime}',
            '--nodes=1', '--ntasks=1',
            f'--output={LOG_DIR}/{label}_%j.log', f'--wrap={wrap}']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--variant', required=True,
                   help="label of the code under test ('main', a PR topic, ...)")
    p.add_argument('--pipe-root', default=REPO_ROOT,
                   help='checkout/worktree whose code runs (default: this one)')
    p.add_argument('--fields', default='', help='comma list (default: all)')
    p.add_argument('--seeds', default='', help="comma list; 0 = clean (default: 0 + the field's seeds)")
    p.add_argument('--cpus', type=int, default=8)
    p.add_argument('--mem', default='48gb')
    p.add_argument('--time', default='03:00:00')
    p.add_argument('--extra', default='', help='extra pipeline args (quoted string)')
    p.add_argument('--env', action='append', default=[], help='K=V exported in the job')
    p.add_argument('--submit', action='store_true')
    p.add_argument('--keep', action='store_true', help='do not clear an existing run dir')
    a = p.parse_args(argv)

    _, fields = RF.load_config()
    names = [n for n in a.fields.split(',') if n] or list(fields)
    env = [tuple(kv.split('=', 1)) for kv in a.env]
    os.makedirs(LOG_DIR, exist_ok=True)
    for name in names:
        spec = fields[name]
        seeds = ([int(s) for s in a.seeds.split(',') if s] if a.seeds
                 else [0] + list(spec['seeds']))
        for seed in seeds:
            cmd = sbatch_command(spec, a.variant, seed, os.path.abspath(a.pipe_root),
                                 cpus=a.cpus, mem=a.mem, walltime=a.time,
                                 extra_args=shlex.split(a.extra), env=env)
            rdir = RF.run_dir(spec, a.variant, seed)
            if not a.submit:
                print(shlex.join(cmd))
                continue
            if (os.path.isdir(rdir) and not a.keep
                    and os.path.basename(rdir).startswith("ref_")):
                shutil.rmtree(rdir)
            jid = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
            print(f'{RF.run_label(name, a.variant, seed)}: job {jid}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
