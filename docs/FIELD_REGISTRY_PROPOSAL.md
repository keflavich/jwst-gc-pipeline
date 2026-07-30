# Proposal: one field registry

**Status:** proposal. `jwst_gc_pipeline/fields.py` is written and tested but
wired into nothing. Supersedes the closed [#98](https://github.com/keflavich/jwst-gc-pipeline/pull/98).

## The problem, measured

Adding a target means editing **seven** places across five files:

| # | registry | where | lines |
|---|---|---|---|
| 1 | `field_to_reg_mapping` | `PipelineRerunNIRCAM-LONG.py`, **inside `__main__`** | 23 |
| 2 | `refnames` | `PipelineRerunNIRCAM-LONG.py` | 53 |
| 3 | `ALIGNMENT_CONFIG` | `alignment_config.py` | 153 |
| 4 | `nvisits` | `crowdsource_catalogs_long.py`, **inside `main()`** | 38 |
| 5 | `obs_filters` | `merge_catalogs.py` (+ a second copy in `make_merged_psf.py`) | 44 |
| 6 | `project_obsnum` | `merge_catalogs.py` | 48 |
| 7 | `offsets_tables` | `merge_catalogs.py`, **inside `main()`** | 15 |

Plus `fov_regname` for MIRI, and the basepath `if target in (...)` branch.

Three live inside functions, so they cannot be imported, inspected, or
overridden — you edit the source. Nothing checks that they agree.

### They already disagree

```
cloudc/2526   in obs_filters, absent from project_obsnum
w51/1182      in project_obsnum, absent from obs_filters
```

`merge_catalogs.py:1482` and `:1701` index `project_obsnum[target][proposal]`
unguarded, so a `cloudc/2526` merge raises `KeyError('2526')`. The `w51/1182`
entry is unreachable — the job list is built from `obs_filters` — so it is
simply dead.

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
    obsid: str | None = None
    nvisits: int | None = None
    filters: tuple[str, ...] = ()
    offsets_table: str | None = None


@dataclass(frozen=True)
class Field:
    name: str
    root: str                     # 'orange' | 'blue'
    observations: tuple[Obs, ...] = ()
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
replaces**, today, exactly. So call sites move one at a time, each a two-line
diff, each independently revertible. The registry is not a new source of truth
until a call site says so.

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
7. Delete the duplicate `obs_filters` in `make_merged_psf.py`.

Steps 2–5 are each small enough to review in one sitting.

## What this does not do

- It does not make the pipeline portable. Paths are still absolute; `root` only
  chooses between the two existing trees. Portability is a separate change,
  which `basepath()` makes tractable by giving it one place to hook.
- It does not fix `cloudc/2526` or `w51/1182`. The registry records them as
  incomplete and dead respectively, and the tests fail loudly if either changes,
  so the fix is a deliberate decision rather than a silent edit.
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
