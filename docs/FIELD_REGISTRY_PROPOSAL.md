# Proposal: one field registry

**Status:** proposal. `jwst_gc_pipeline/fields.py` is written and tested but
wired into nothing. Supersedes the closed [#98](https://github.com/keflavich/jwst-gc-pipeline/pull/98).

## The problem, measured

Adding a target means editing **nine** registries across six files. Seven you
hit directly:

| # | registry | where | lines |
|---|---|---|---|
| 1 | `field_to_reg_mapping` | `PipelineRerunNIRCAM-LONG.py`, **inside `__main__`** | 23 |
| 2 | `refnames` | `PipelineRerunNIRCAM-LONG.py` | 53 |
| 3 | `ALIGNMENT_CONFIG` | `alignment_config.py` | 153 |
| 4 | `nvisits` | `crowdsource_catalogs_long.py`, **inside `main()`** | 38 |
| 5 | `obs_filters` | `merge_catalogs.py` (+ a second copy in `make_merged_psf.py`) | 44 |
| 6 | `project_obsnum` | `merge_catalogs.py` | 48 |
| 7 | `offsets_tables` | `merge_catalogs.py`, **inside `main()`** | 15 |

And two more, found while reviewing this proposal: `obs_filters_niriss`
(`merge_catalogs.py:93`, silently substituted by `_obs_filters_for`) and
`obs_ids` (`make_merged_psf.py`). Plus `fov_regname` for MIRI and the basepath
`if target in (...)` branch.

Three live inside functions, so they cannot be imported, inspected, or
overridden — you edit the source. Nothing checks that they agree.

### They already disagree

```
cloudc/2526   in obs_filters, absent from project_obsnum
w51/1182      in project_obsnum, absent from obs_filters
```

`merge_catalogs.py:1482` indexes `project_obsnum[target][proposal]` unguarded,
so a `cloudc/2526` merge raises `KeyError('2526')` — reproduced. (`:1701` is
*not* reachable for that pair; an earlier `continue` fires first. An earlier
draft of this document claimed both.) The `w51/1182` entry is unreachable — the
job list is built from `obs_filters`, and the one other reader guards on
membership — so it is simply dead.

A third instance of the same class, found while reviewing this proposal:
`offsets_tables` omits `1905`, `3523` and `2526` entirely, so
`offsets_tables[progid]` raises `KeyError` for **every wd1 and wd2 per-filter
merge**.

Neither is exotic. They are what happens when the same fact is written down
three times by hand.

### And one is transposed

`obs_filters` and `project_obsnum` are `{target: {proposal: …}}`.
`nvisits` is `{proposal: {target: …}}`. Same twenty (target, proposal) pairs,
opposite nesting, and the reader has to hold that in their head.

## The proposal

One dataclass per field, one per observation, in `jwst_gc_pipeline/fields.py`:

```python
@dataclass(frozen=True)
class Obs:
    proposal: str
    obsid: Optional[str] = None
    nvisits: Optional[int] = None
    filters: Tuple[str, ...] = ()
    offsets_table: Optional[str] = None


@dataclass(frozen=True)
class Field:
    name: str
    root: str                     # 'orange' | 'blue'
                                  # typing.Optional, not `X | None`: the
                                  # declared floor is Python 3.9
    observations: Tuple[Obs, ...] = ()
```

The whole registry is **76 lines** for 17 fields and 20 observations:

```python
Field('brick', root='blue', observations=(
    Obs('2221', obsid='001', nvisits=2,
        filters=('f410m', 'f212n', 'f466n', 'f405n', 'f187n', 'f182m', 'f2550w')),
    Obs('1182', obsid='004', nvisits=2,
        filters=('f444w', 'f356w', 'f200w', 'f115w')),
)),
```

Adding a target becomes one block, in one file, that cannot be internally
inconsistent — a missing `obsid` is visibly missing rather than a `KeyError`
three modules away.

## Why this can land safely

**Views, not a rewrite.** The module exposes each existing dictionary as a
derived function:

```python
fields.obs_filters()      # {target: {proposal: [filters]}}
fields.project_obsnum()   # {target: {proposal: obsid}}
fields.nvisits()          # {proposal: {target: n}}   <- transposition is the view's job
fields.basepath(target)   # replaces the if/else branch
```

`test_fields_registry.py` asserts each view **equals the dictionary it
replaces**, today, exactly — including **key order**, which is not cosmetic:
`individual_frame_merge_jobs` derives the SLURM array index from each target's
proposal order, so a reordering would silently send array tasks at the wrong
filter. So call sites move one at a time, each independently revertible, and the
registry is not a source of truth until a call site says so.

The registry data was **generated from the current literals**, not retyped, so
it is faithful by construction.

## Migration, in order

1. **This PR.** Add `fields.py` + equivalence tests. Nothing imports it. No
   behaviour change.
2. `merge_catalogs` reads `obs_filters` and `project_obsnum` from the views;
   delete the literals. Equivalence tests keep proving the swap is inert.
3. `crowdsource_catalogs_long` reads `nvisits` from the view; the dict leaves
   `main()`.
4. `basepath()` replaces the `if target in (...)` branch — this also gives the
   MIRI and NIRISS drivers a basepath they can honour, which is the missing
   piece for running off HiPerGator.
5. Fold in `field_to_reg_mapping`, `refnames`, `fov_regname`.
6. `ALIGNMENT_CONFIG` last, and possibly never — it is already a single typed
   registry with its own tests. A `Field` could carry a reference to its
   `FieldAlignment` rather than absorbing it.

   **`refnames` needs care, and is not revertible once folded in.** It is keyed
   by *proposal*, not (field, proposal): `refnames['2221']` is one value shared
   by brick and cloudc. Storing it per-`Obs` silently widens that to per-field
   and lets the two diverge. Its `'THIS_IS_A_BUG_IF_YOU_USE_THIS'` sentinel also
   has no home in `Obs`. Same for `ALIGNMENT_CONFIG`.
7. `make_merged_psf.py` last. It is **not** a duplicate to delete: it is
   proposal-keyed, uppercase, brick-only, missing `f2550w`, and carries its own
   `obs_ids`. That step is a rewrite.

Steps 2–4 are each small enough to review in one sitting. Step 2 is **not** the
two-line diff an earlier draft of this document claimed: the real accessor is
`_obs_filters_for()`, which substitutes `obs_filters_niriss` for NIRISS runs —
an eighth registry that `Field`/`Obs` cannot represent as written, because
NIRISS reuses NIRCam filter names. Either `Obs` grows an instrument, or the
NIRISS set stays where it is and the view composes with it.

## What this does not do

- It does not make the pipeline portable. Paths are still absolute; `root` only
  chooses between the two existing trees. Portability is a separate change,
  which `basepath()` makes tractable by giving it one place to hook.
- It does not fix `cloudc/2526`, `w51/1182`, or the missing `offsets_tables`
  proposals. The registry records `cloudc/2526` as incomplete and `w51/1182` as
  dead, and covers the missing offsets proposals by construction; tests fail
  loudly if any of it changes, so each fix stays a deliberate decision.
- It does not touch `ALIGNMENT_CONFIG`'s content.

## Open questions

1. Should `Field.root` be a path rather than `'orange'`/`'blue'`? A path is
   more honest but bakes the absolute tree into the registry; the enum leaves
   one place to intercept.
2. Should the registry be data (YAML/TOML) instead of Python? Data cannot carry
   the comments that explain *why* cloudef is two observations of one field,
   and those comments are load-bearing.
3. Is `offsets_table` a registry fact or a runtime lookup? It is currently a
   path read at merge time; the registry could hold the path and let the caller
   read it, which is what the lazy read added in #219 already does.
