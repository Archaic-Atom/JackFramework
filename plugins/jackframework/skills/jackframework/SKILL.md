---
name: jackframework
description: >-
  How to build on, launch, and debug JackFramework (JF) — a PyTorch training
  framework that owns the training loop, DDP launcher, checkpoint I/O and progress bar
  while you supply a model-interface class, a dataloader class and a launcher. Use this
  whenever a task touches JF hooks (inference / loss / accuracy / post_process /
  get_train_dataset / save_result), launching `python Source/main.py --mode train|test`,
  JF CLI flags (--modelDir --modelName --dataset --dist --gpu --imgNum --dataloaderType),
  checkpoints and checkpoint.list, warm-starting from a prior .pth, or any JF run that
  failed. Trigger it even when JF is never named — e.g. "training silently skips my loss
  function", "dies with an empty ProcessRaisedException", "the progress bar never reaches
  100%", "checkpoint list file not found", "the eval finished but there's no output csv",
  "how do I resume from epoch 4". Most JF failures are silent, or put the traceback
  nowhere near the real cause, so consult this before guessing at a fix.
---

# JackFramework (JF)

JF is a training framework built around string dispatch and a per-project config.
The deal it offers: it owns the **training loop, DDP launcher, checkpoint I/O,
logging, progress bar and resume**, and in exchange you implement a **model-interface
class**, a **dataloader class**, and a thin launcher script.

That deal has one sharp edge that causes most of the pain: **JF finds your code by
string name via `getattr`, with no abstract base class and no validation.** A method
you spelled slightly wrong is not an error — it is silently never called. Keep that
in mind and most JF debugging becomes tractable.

The framework is installed separately from the project that uses it. Confirm which
copy is actually imported before debugging framework behavior — a stale `.egg` next
to a source checkout is a classic time sink:

```bash
python -c "import JackFramework; print(JackFramework.__path__)"
pip show JackFramework          # 'Editable project location' if installed -e
```

Use `__path__`, not `__file__`: for a namespace-style install `__file__` is `None`.

## The user-side contract

Everything below is dispatched by name. Copy signatures verbatim from a working
example rather than retyping them.

**Model interface** (subclass of `ModelHandlerTemplate`):

| hook | contract |
|---|---|
| `inference(model, input_data, model_id)` | **returns a LIST** — `[output_dict]` — even with a single model |
| `loss(output_data, label_data, model_id)` | `output_data` is that LIST; unwrap with `output_data[self.ID_PRED]` (=`[0]`). Returns `[total, *components]` |
| `accuracy(output_data, label_data, model_id)` | same list-unwrap; returns `[metrics]` |
| `post_process(epoch, rank, ave_tower_loss, ave_tower_acc)` | **underscore**, not `postprocess`. Once per epoch (train) / after the test loop |
| `pretreatment(epoch, rank)` | pre-epoch hook |
| `load_model(model, checkpoint, model_id) -> bool` | often `strict=False` for partial loads / variant inheritance |

**Dataloader** (subclass of `DataHandlerTemplate`):

| hook | contract |
|---|---|
| `get_train_dataset(path, is_training=True)` | also called with `is_training=False` for the test set — the name is misleading |
| `save_result(output_data, supplement, img_id, model_id)` | for `--mode test`; same list-unwrap as loss/accuracy |

Register the dataloader class in the project's dataloader zoo dict; the key is what
`--dataset` matches, just as `--modelName` matches the model zoo. A missing key
surfaces as a bare `AssertionError` with no hint, so register before launching.

The list-wrapping rule is the second most common silent failure: `inference()` returns
`[output]`, but downstream hooks receive that *list*, not the dict. If a check like
`'<key>' in output_data` is always False, you forgot `output_data[self.ID_PRED]`.

## Launching

There is one entry point — `python Source/main.py <flags>` — with `--mode` selecting
behavior. JF's own flags are:

```
--mode --modelName --dataset --dist --gpu --port --ip --nodes --node_rank
--batchSize --lr --maxEpochs --sampleNum --auto_save_num --pretrain --debug
--trainListPath --valListPath --imgNum --valImgNum --imgWidth --imgHeight
--size_magnification --dataloaderNum --dataloaderType
--modelDir --outputDir --resultImgDir --log --web_cmd
```

Anything else on a launcher's command line was added by that project's
`user_interface.py`. Worth checking before assuming a flag is framework behavior —
projects commonly add their own architecture/loss switches.

Conventions worth internalizing:

- `--gpu` is a **count**, not a device list. Select devices with `CUDA_VISIBLE_DEVICES`.
- `--port` is the DDP rendezvous port — vary it across concurrent runs or they collide.
- `--dataloaderType` defaults to `mapstyle`. An iterable/WebDataset pipeline needs it
  passed **explicitly**.
- `--imgNum` does **not** limit how much data is processed — it drives the progress bar
  (`imgNum / batchSize`, per rank under DDP). Everything the dataloader yields is
  processed either way: a map-style run walks the whole list, and an iterable run
  drains each rank's shard share until `StopIteration`. So a bar reading `64/64` can
  have processed every one of 2000 samples.

  **Set it to the real total sample count anyway.** It is the one number that makes
  the bar and the ETA meaningful, and a wrong value is not always cosmetic: if the
  project's dataloader also applies a length cap derived from it (see pitfall 5), a
  too-large value can make the epoch never end, so auto-save never fires. Treat a bar
  that cannot reach 100% as a real alarm.
- On a config-driven project, some architecture/loss flags are **inert** — the config
  file wins. Launchers often pass them anyway "for parity", which misleads readers.
  Check the config before believing a flag did anything.

## Checkpoints

`--modelDir` accepts **either** form:

- **a directory** — JF reads `checkpoint.list` inside it to pick the file. Without that
  file you get `Checkpoint list file not found` and nothing loads.
- **a `.pth` file** — loaded directly. New checkpoints are then written **alongside
  it**, into its parent directory.

The file form needs a JF new enough to include the `--modelDir`-takes-a-file change.
On older copies the run dies during init at `FileHandler.mkdir` with `Unable to create
directory because a file already exists`, because bootstrap tried to `mkdir` the file
name — the traceback names `init_handler.py`, which reads as unrelated to checkpoints.

`checkpoint.list` is two plain lines, and hand-writing it is a normal, supported move
when you want to pin one exact epoch:

```
last model name:model_epoch_4.pth
model_epoch_4.pth
```

Saving writes `model_epoch_<N>.pth` every `--auto_save_num` epochs into `--modelDir`
and puts the newest at the top of that list, so the **first** line is what loads next —
not necessarily the highest-numbered file in the directory.

Projects often save only a sub-module's `state_dict` (e.g. a trainable head, not a
frozen backbone). So a checkpoint that loads with `missing=0, unexpected=0` against a
*different* variant is expected when the variants share that sub-module — it is not
by itself evidence that you loaded the right thing. Verify against the log line naming
the loaded file.

To **warm-start a fresh run** from a prior checkpoint, strip the optimizer state and
reset the epoch counter first; otherwise JF resumes the stored epoch count and your
new run's schedule and warmup are computed from the wrong epoch.

## Debugging recipe

1. **Reproduce on a single process first** — `--dist False --gpu 1`. This is the single
   highest-value habit: DDP swallows tracebacks, single-process shows the real error.
2. **Empty `ProcessRaisedException` with no Python traceback** is almost always DDP
   hitting a registered parameter that received no gradient — a frozen backbone or
   encoder is the usual cause. It looks like a C-level crash but it isn't. Fix by
   genuinely setting `requires_grad=False` before the model is wrapped, or with
   `find_unused_parameters=True` in JF's DDP wrap.
3. **A hook seems to do nothing** → suspect the name before the logic.
4. **`--mode test` finished but there's no output CSV** → the step that merges the
   per-rank temp files runs after the loop and can lag or hang. No data is at risk:
   predictions are already in `<outputDir>/.tmp_test_*_rank<N>.csv` (leading dot — plain
   `ls` hides it). Read or concatenate those rather than re-running.
5. **Progress bar never reaches 100% and auto-save never fires** → a per-worker length
   cap applied as if it were global. See pitfall 5.

For the full symptom→cause catalog, read `references/pitfalls.md`.

## Working style

When something breaks, prefer a **surgical patch** in the project over changing the
framework, and keep framework patches minimal and recorded — a JF patch lives in the
JF install rather than the project's version control, so a reinstall silently reverts
it and the next person re-diagnoses from scratch.

When you do patch JF itself, verify the change by *running* it, not by reading it. The
init and checkpoint paths in particular have ordering that is hard to predict from
source — the mkdir-before-validate ordering described above is exactly the kind of
thing that reading gets wrong.
