# JackFramework skill for Claude Code

A [Claude Code](https://claude.com/claude-code) skill that teaches the assistant how
JackFramework actually behaves — so it stops guessing when a run fails.

JF is built around string dispatch and a per-project config: it owns the training loop,
DDP launcher, checkpoint I/O, logging and progress bar, while you supply a
model-interface class, a dataloader class and a launcher. That design has a sharp edge —
**hooks are resolved by name via `getattr`, with no base-class enforcement**, so a
misspelled method is silently never called rather than raising. Several of JF's other
failure modes are similarly quiet, or produce a traceback pointing somewhere unrelated
to the real cause.

This skill encodes that knowledge: the hook contract, the CLI flags that are genuinely
JF's (versus ones a project added), checkpoint and `checkpoint.list` semantics, and a
symptom → cause → fix catalog.

## What it covers

| | |
|---|---|
| **Hook contract** | `inference` / `loss` / `accuracy` / `post_process` / `pretreatment` / `load_model`, `get_train_dataset` / `save_result` — signatures, the list-wrapping rule, and the silent-dispatch trap |
| **CLI** | JF's own flags, and the conventions that surprise people (`--gpu` is a count; `--dataloaderType` defaults to `mapstyle`; `--imgNum` drives the progress bar rather than limiting the data) |
| **Checkpoints** | `--modelDir` as either a directory with `checkpoint.list` or a `.pth` file; the two-line list format; which line actually loads; warm-starting from a prior run |
| **Debugging** | Why an empty `ProcessRaisedException` means frozen parameters, why a test run can finish without its merged CSV, why a progress bar that can't reach 100% means auto-save will never fire |

## Install

### As a plugin (gets updates)

```
/plugin marketplace add Archaic-Atom/JackFramework
/plugin install jackframework
```

### As a plain skill (copy it)

The skill is a self-contained directory, so it also works by copying:

```bash
git clone https://github.com/Archaic-Atom/JackFramework.git
cp -r JackFramework/plugins/jackframework/skills/jackframework ~/.claude/skills/
```

Use `.claude/skills/` inside a project instead of `~/.claude/skills/` if you want it
scoped to one repository.

## Using it

Nothing to invoke — the skill declares when it applies, and Claude consults it when a
task involves JF. In practice it triggers on things like:

- "my training runs but the loss function never gets called"
- "4-GPU training dies with ProcessRaisedException and no traceback"
- "the eval finished but there's no csv in the output dir"
- "how do I evaluate just one epoch's checkpoint"

You can also load it explicitly with `/jackframework`.

## Scope

The skill describes the framework, not any particular project built on it. Where
behavior depends on the project (config-driven architecture flags, the dataloader zoo,
custom CLI flags added in `user_interface.py`), it says so and tells the assistant to
check the project rather than assume.

Framework behavior changes over time. The one version-sensitive item called out in the
skill is `--modelDir` accepting a `.pth` file directly, which older checkouts do not
support — on those, the run fails during init at `FileHandler.mkdir`.
