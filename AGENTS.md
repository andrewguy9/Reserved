# Working on reserved

Use these checklists to prepare, implement, and review a change. Complete the
applicable checks with evidence. Mark a check not applicable only when its
condition does not apply; an unverified guarantee remains unverified.
Keep the working checklist in the task or PR, not by ticking this shared file.
For a small change, keep the plan and report brief.

Reserved provides single-machine, cooperative resource coordination. Its
correctness depends on both allocation policy and OS resource mechanics.
Keep policy separate from mechanics, make ownership explicit, and give
consumers a simple lifetime model.

## 1. Before editing

- [ ] Read the relevant implementation and tests, plus `README.md` and
  `ARCHITECTURE.md`. Treat `RESERVED_DESIGN.md` and `DAEMON_REFACTOR_PLAN.md`
  as historical context, not proof of current behavior.
- [ ] State the requested behavior, the observable result, and the guarantees
  that must remain true.
- [ ] Identify the affected layers using the module map below.
- [ ] Define a bounded change and the checks that will establish correctness.
- [ ] Inspect the working tree and preserve existing user changes.
- [ ] Resolve any ambiguity that materially changes the public contract or
  scope. Proceed with routine implementation choices without extra approval.

If code and documentation disagree, explain a discrepancy relevant to the
work. Discovering an architectural exception does not authorize fixing it as
part of an unrelated task. If a much larger redesign becomes necessary,
explain why and establish the new scope before expanding the change.

## 2. Check the product guarantees

For changes affecting reservation behavior, verify the applicable guarantees:

- [ ] Acquisition is atomic: all reservations in a request are granted or
  none are. Rejection leaves accounting unchanged.
- [ ] Reservations acquired together release independently; releasing one
  returns only its claimed capacity.
- [ ] A reservation remains alive while any client lease copy is open,
  including inherited or duplicated copies. The daemon reclaims capacity
  after observing the last client copy close, including after exit or crash.
- [ ] Daemon restart invalidates leases; SDK `check()` exposes the loss.
  Previous reservations are not silently restored.
- [ ] The daemon accepts explicit device assignments. Constraint solving stays
  in clients and outside the daemon protocol.
- [ ] The daemon checks current capacity even if a client solver found room
  in an earlier status snapshot.
- [ ] The change preserves immediate grant or rejection, in-memory state, and
  cooperative participation. It does not incidentally add queuing,
  enforcement, persistence, or multi-machine coordination.

A requested change to these guarantees is a design change: make its
consequences explicit.

## 3. While implementing

- [ ] Keep allocation policy and logical transitions separate from socket
  operations, handle cleanup, and presentation. Do not duplicate allocation
  policy in the daemon shell or SDK.
- [ ] Give new policy computations explicit inputs. Do not implicitly read
  time, randomness, environment variables, or network state. Existing time
  and ID generation in `core.grant()` is an exception, not an instruction to
  expand unrelated work.
- [ ] Parse external values at config, protocol, and CLI boundaries and
  establish their domain invariants. Type annotations alone do not validate
  runtime input.
- [ ] Use typed domain records in new core code. Preserve meaningful
  distinctions between missing, empty, zero, and failed. Do not use `None`
  or an empty collection to hide an error.
- [ ] Keep computations small and dependencies explicit. Use comprehensions,
  loops, closures, or factories where they make the operation clearer.
  Avoid flags selecting unrelated operations and speculative abstractions.
- [ ] Preserve caller-owned inputs where value transformations are promised.
  Check nested containers: a frozen dataclass does not freeze its dictionaries.
- [ ] Preserve SDK signatures, CLI syntax, environment variables, wire formats,
  and device-index meanings unless changing them is part of the task.
- [ ] Keep acquisition, use, release, and failure behavior understandable
  through the public API without exposing transport machinery.
- [ ] Keep one coherent idea in the diff. Exclude unrelated cleanup, renaming,
  dependencies, and future features. A helper and its first use can land
  together when they form one understandable change.

IO is essential to Reserved. Structure it to make its guarantees clear;
reducing the effects layer's line count is not a goal by itself.

## 4. When touching sockets, descriptors, or process lifetimes

- [ ] Identify the owner and closer of each handle, including every point
  where ownership transfers or a duplicate is created.
- [ ] Follow failure paths between capacity allocation, lease creation, and
  descriptor transfer. Ensure cleanup cannot leak handles or strand capacity.
- [ ] Close received descriptors that cannot be accepted or used.
- [ ] Preserve intentional lease inheritance across `exec`, prevent unrelated
  inheritance, and account for duplicate handles across `fork`.
- [ ] Handle partial stream reads and writes. Check ancillary-data truncation
  and descriptor counts where descriptors are transferred.
- [ ] Distinguish malformed input, insufficient capacity, transport failure,
  and expired leases. Avoid broad catches that hide these differences.
- [ ] Keep cleanup close to ownership. Make repeated release safe where the
  public interface promises it.
- [ ] When placement or execution changes, verify physical CUDA ordinals
  against logical indices inside `CUDA_VISIBLE_DEVICES`.

## 5. Verify before handoff

Choose tests from the guarantee being changed. Mocks can exercise errors;
they cannot prove descriptor transfer, inheritance, or process-exit cleanup.

- [ ] For a bug fix, add a regression test that exposes the faulty behavior.
- [ ] For changed computations, test small inputs with explicit expected
  values, including relevant boundaries and failure cases.
- [ ] For changed state transitions, check returned effects, accounting, and
  unchanged input state where immutability is promised.
- [ ] For changed OS mechanics, exercise the affected guarantee with real
  sockets, descriptor passing, or subprocesses. Include the affected failure
  and cleanup paths. Configured resource accounting and leases need no GPU.
- [ ] For a behavior-preserving refactor, compare observable results and
  exercise the mechanics affected by the change.
- [ ] Assert contracts rather than incidental private call sequences. Use
  bounded waits and observable conditions rather than arbitrary sleeps.
  Isolate test sockets and processes from the user's running daemon.
- [ ] Run focused checks while working. After the final code change, run
  `make check` in the project environment with development dependencies.
- [ ] Inspect the final diff for unintended changes and whitespace errors.
- [ ] Record commands, results, and material verification gaps. A skipped or
  blocked check does not count as passed. Name any platform-specific behavior
  that remains unverified.

For documentation-only changes, verify the claims, links, and diff; the code
suite is not required. `make check` runs `python3 -m pytest tests/ -v`.
`make coverage` and `make htmlcov` are available when useful. No type-checker
or linter gate is currently configured; do not claim checks you did not run.

## 6. Make the result reviewable

- [ ] Explain the problem, resulting behavior, important design choices, and
  verification evidence in the handoff or PR description.
- [ ] Point reviewers to ownership or accounting logic requiring careful
  inspection. State remaining limitations plainly.
- [ ] Document intentional public behavior changes with an example.
- [ ] Retain reusable diagnostic tools and generators in the repository with
  instructions and appropriate verification. Preserve anything needed to
  reproduce the delivered result; disposable inspection commands need not
  become maintained tools.
- [ ] If review exposes a recurring failure, record the Reserved incident and
  evidence in `LEARNINGS.md` and refine the corresponding rule here. Create
  that file when there is an incident to record; do not import another
  project's history as Reserved's evidence.

Work is ready when its applicable checks have evidence, the code is clearly
shaped, and the change is small enough to review. Report incomplete work
honestly rather than treating a checklist as proof.

## Module map

| Module | Responsibility |
| --- | --- |
| `reserved/core.py` | Allocation policy and resource accounting |
| `reserved/daemon_state.py` | Logical events, transitions, and effect descriptions |
| `reserved/daemon.py` | Sockets, descriptor ownership, effect execution, process integration, monitoring IO |
| `reserved/protocol.py` | Conversion between wire messages and domain values |
| `reserved/sdk/solvers.py` | Placement decisions from a status snapshot |
| `reserved/sdk/client.py` | Acquisition, lease ownership, public Python interface |
| `reserved/bin/` | CLI parsing, presentation, execution |
