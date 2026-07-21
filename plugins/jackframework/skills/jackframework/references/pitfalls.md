# JF pitfall catalog — symptom → cause → fix

Ordered roughly by how often each one bites. Every entry starts from the *symptom*,
because that is what you actually have when you arrive.

## Contents

1. [A hook never runs](#1-a-hook-never-runs)
2. [`'<key>' not in output_data` forever](#2-key-not-in-output_data-forever)
3. [Empty `ProcessRaisedException` on multi-GPU](#3-empty-processraisedexception-on-multi-gpu)
4. [Bare `AssertionError` at startup](#4-bare-assertionerror-at-startup)
5. [Progress bar never reaches 100%, no auto-save](#5-progress-bar-never-reaches-100-no-auto-save)
6. [Checkpoint didn't load / loaded the wrong epoch](#6-checkpoint-didnt-load--loaded-the-wrong-epoch)
7. [`--mode test` produced no output CSV](#7---mode-test-produced-no-output-csv)
8. [A CLI flag appears to be ignored](#8-a-cli-flag-appears-to-be-ignored)
9. [Editing JF has no effect](#9-editing-jf-has-no-effect)
10. [Indices collide in saved results](#10-indices-collide-in-saved-results)

---

## 1. A hook never runs

**Symptom** — training proceeds but your loss/metric/post-processing clearly never
executes. No error anywhere.

**Cause** — JF dispatches user methods by string name with `getattr`. There is no ABC
and no validation, so a misspelling is indistinguishable from "the user didn't
implement this hook". The classic is `postprocess` vs the correct `post_process`; wrong
case and singular/plural slips do the same thing. Because the template base class
supplies a no-op default, the call still succeeds — it just runs the empty parent.

**Fix** — diff your method names against a working example rather than against memory.
If you want a guardrail, assert on `hasattr` in your own `__init__`.

## 2. `'<key>' not in output_data` forever

**Symptom** — a downstream hook insists a key you definitely produced is missing.

**Cause** — `inference()` returns a **list** (`[output_dict]`). `loss()`, `accuracy()`
and `save_result()` receive that list, not the dict inside it.

**Fix** — unwrap first: `pred = output_data[self.ID_PRED]` (`ID_PRED` is `0`). This
applies to every hook downstream of `inference`, which is why it tends to be fixed in
one place and forgotten in another.

## 3. Empty `ProcessRaisedException` on multi-GPU

**Symptom** — a DDP run aborts with `ProcessRaisedException` and no Python traceback.
Looks like a C-level crash.

**Cause** — a registered parameter received no gradient, so the reducer aborted in C++
rather than raising a Python exception, leaving nothing to propagate back. In practice
this means part of the model is frozen while still being registered with DDP.

**Fixes**, in order of preference:

1. **Freeze properly and early** — set `requires_grad=False` on the frozen module when
   the model is built, before JF wraps it. DDP snapshots which parameters expect
   gradients at construction time, so freezing later (e.g. inside `pretreatment()`) is
   too late to matter.
2. **`find_unused_parameters=True`** in JF's DDP wrap. This costs a little per-step
   overhead but also makes the *real* error surface if the cause was something else.
3. **`static_graph=True`** if the graph is structurally constant, as a cheaper
   alternative to (2).

To locate the offending parameters, run one process and list what has
`requires_grad=True` but a `None` gradient after a backward pass. Also useful:
`TORCH_DISTRIBUTED_DEBUG=DETAIL` makes PyTorch name them for you.

**Always confirm on `--dist False --gpu 1` first** — single-process gives a real
traceback, so you learn whether you even have a DDP problem.

A JF-side patch such as (2) lives in the JF install, not the project's version control,
so a reinstall silently reverts it. Check whether the patch is still present before
re-diagnosing.

## 4. Bare `AssertionError` at startup

**Symptom** — an assertion fires before training starts, with no message.

**Cause** — a registry lookup missed: `--dataset` has no matching key in the dataloader
zoo, or `--modelName` has none in the model zoo.

**Fix** — grep the zoo dict and compare the exact string. Register new variants *before*
launching; JF will not tell you which key it wanted.

## 5. Progress bar never reaches 100%, no auto-save

**Symptom** — the bar stalls short of the end, and because the epoch never "completes",
`--auto_save_num` never triggers, so a long run produces zero checkpoints. This failure
is expensive precisely because it looks like slow progress rather than a bug — it can
burn a multi-day run before anyone notices there are no saves.

**Cause** — a per-worker length cap applied as if it were global. With WebDataset,
`with_epoch(N)` yields N samples **per worker**, so the real epoch is
`world_size × num_workers × N` — e.g. ~32× over-iteration on a 4-GPU / 8-worker setup.
The bar hits 100% (it is drawn from `imgNum / batchSize`) and the loop just keeps going,
so the epoch never ends and no save fires.

**Fix** — the settled answer is to **remove `with_epoch` entirely** and let
`split_by_node` + `split_by_worker` partition the shards, so each worker drains its own
share once and `StopIteration` propagates naturally. Dividing the cap by `world_size`,
or by `world_size × num_workers`, are both still wrong (the latter under-counts whenever
a rank has fewer shards than workers, leaving some workers idle).

Then set `--imgNum` to the **true total sample count** so the bar means something. Under
DDP the first epoch's bar can still read `world_size`× too high; JF self-corrects on
later epochs and logs "input images numbers is different the number of datasets" once.
That single log line is expected — it is not the bug.

## 6. Checkpoint didn't load / loaded the wrong epoch

**Symptom** — `Checkpoint list file not found`, or the run silently starts from scratch,
or it resumes an epoch you didn't intend.

**Cause and fix** — depends on what `--modelDir` points at:

- **A directory without `checkpoint.list`** → nothing loads. Write the two-line file:
  ```
  last model name:model_epoch_4.pth
  model_epoch_4.pth
  ```
  The **first** line is what gets loaded (JF reads row 1 and strips the
  `last model name:` prefix); entries below it are history. So the file that loads is
  not necessarily the highest-numbered `.pth` in the directory — pinning one exact epoch
  by hand is normal practice for evaluation.
- **A `.pth` file** → loaded directly; new saves land in its parent directory. Requires a
  JF new enough to include the `--modelDir`-takes-a-file change. On an older copy the run
  dies in init at `FileHandler.mkdir` with `Unable to create directory because a file
  already exists` — the traceback names `init_handler.py`, which is why this reads as
  unrelated to checkpoints.
- **Resuming an unintended epoch** → the checkpoint still carries its optimizer state and
  epoch counter. For a *fresh* run warm-started from old weights, strip both first,
  otherwise the schedule and any warmup are computed from the wrong epoch.

`missing=0, unexpected=0` when loading across two variants is **expected** if they share
the saved sub-module — many projects save only a trainable head rather than the whole
model. It is not by itself evidence that you loaded the right thing; confirm against the
log line naming the file that was loaded.

A related silent failure: `load_model()` is often `strict=False`, so a `module.` prefix
mismatch (a checkpoint trained under DDP, evaluated with `--dist False`) can drop every
key and still produce a complete, plausible-looking run on random weights. Check the
missing/unexpected counts, not just that the run finished.

## 7. `--mode test` produced no output CSV

**Symptom** — the progress bar hit 100%, the process is idle or was killed, and the
merged `test_<dataset>_<timestamp>.csv` is missing.

**Cause** — the step that merges the per-rank temp files runs after the loop and can lag
by minutes, or hang.

**Fix** — nothing is lost. Every prediction is already in
`<outputDir>/.tmp_test_*_rank<N>.csv` (note the leading dot — plain `ls` hides it). Read
that file directly, or concatenate the per-rank ones keeping one header. Re-running the
eval to "get the CSV" wastes GPU time for no new data.

If the temp files are genuinely absent, the failure is upstream instead: the test-mode
hooks must line up — `inference()` attaching the output, `save_result()` unwrapping the
list and writing, and `post_process()` doing any final merge. A misspelled hook (pitfall
1) or a missing list-unwrap (pitfall 2) produces exactly this.

## 8. A CLI flag appears to be ignored

**Symptom** — you pass a flag and nothing changes.

**Cause** — either the flag is not a JF flag at all (projects add their own in
`user_interface.py`), or on a config-driven project the config file is authoritative for
architecture/loss values and those CLI flags are inert. Launcher scripts frequently pass
them anyway for parity with older scripts, which makes them look load-bearing.

**Fix** — check the project's config before believing a flag. Be aware of the inverse
trap too: when the same quantity exists in both places (a dataloader-side flag and a
config-side field), nothing cross-checks them, so they must be kept consistent by hand.

## 9. Editing JF has no effect

**Symptom** — you patch a JF source file and the behavior doesn't change.

**Cause** — JF often exists as several copies: a built `.egg` in site-packages, the
source checkout, and stale `build/` artifacts. You edited one that isn't imported.

**Fix** —
```bash
python -c "import JackFramework; print(JackFramework.__path__)"
pip show JackFramework    # look for 'Editable project location'
```
Prefer an editable install of the source for anything you intend to debug. Use
`__path__`, not `__file__` — the latter is `None` for a namespace-style install.

## 10. Indices collide in saved results

**Symptom** — rows in the output overwrite each other, typically near the end of a run.

**Cause** — a global index computed as `img_id * batch + b` using the *actual* tensor
size. The final batch is usually partial, so its smaller size shifts every index and
collides with earlier full batches.

**Fix** — compute the stride from the **configured** batch size (`args.batchSize`), not
from `tensor.shape[0]`.
