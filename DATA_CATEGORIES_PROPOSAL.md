# Resources and Metadata: Data Categories for Devices

## Executive summary

Reserved coordinates hardware on one machine through leases. A process claims
capacity, holds a file descriptor, and the daemon reclaims the capacity when
the descriptor closes. We want the same mechanism to bound parallel requests
to a service: a hosted model API or a local inference server. Designing that
one feature showed that the lease model already fits and the data model does
not. This document proposes three steps.

1. **Data categories.** A device becomes a free-form `name`, a map of
   `resources` that obey semaphore rules, and a map of immutable scalar
   `metadata`. The daemon's accounting becomes generic over resource names,
   every value is validated at the config and wire boundaries, and each grant
   carries the device's metadata so a worker can build its CUDA mask from the
   grant alone. This step changes the config, request, grant, and status
   formats once, pre-1.0.
2. **Constraint language and solver.** In the SDK and CLI only: per-reservation
   variables, metadata constraints, resource claims, groups that are required
   or optional, and a backtracking solver that plans against one snapshot and
   emits a sequence of atomic requests.
3. **The service feature.** A `service` kind with a `concurrency` resource.
   Expected to need no core, protocol, or solver change, which is the test
   that steps one and two were shaped correctly.

Throughout, the daemon stays a subtraction machine that grants or rejects one
explicit request atomically. Everything that looks like judgment lives in the
solver, which proposes what to attempt.

## The problems we encountered

The shape of this proposal is the sum of six problems met in order while
trying to add the service feature. Each resolution is a design rule the rest
of the document applies.

**1. A third hard-coded resource.** The service feature needs a count of
simultaneous requests. Reusing `compute_units` for it was rejected because the
name is unclear. Adding a `concurrency` field would mean touching `Device`,
`DeviceAllocation`, `DeviceStatus`, `DeviceConstraint`, both solvers, both
CLIs, the dashboard, and the wire, and doing it again for every resource after
that. *Rule:* resource accounting is generic over names. Code must not depend
on `ram_mb` or `compute_units`.

**2. Generic over resources is not ignorant of kind.** The first attempt to
make everything generic ran into the parts of the system that must know what a
CUDA card is: the `CUDA_VISIBLE_DEVICES` mask, the torch device lookup, and
UUID validation. Those are mechanics, not accounting. *Rule:* two data
categories. Resources are counted; metadata is matched. `kind` is metadata,
authoritative for the mechanics layer, and the only metadata key the daemon
reads.

**3. Facts the model cannot express.** Three requests had nowhere to live:
two cards whose total memory is over 24 GB, two cards on the same NVLink
network, and a host power budget shared by every card. The first is a fact
about a card, which today's `ram_mb=N` conflates with a claim on free
capacity. The second is a relation between devices. The third is capacity on
a pool rather than a device. *Rules:* metadata holds facts; a constraint
language in the solver expresses relations such as `a.nvlink == b.nvlink`;
a pool is an ordinary device, claimed in the same atomic reservation as the
cards that draw from it. None of these needs hierarchy in the ledger.

**4. Lenient parsing.** `int(obj.get(key, 0))` accepts `-1`, `"5"`, and `2.7`,
turns `null` into a `TypeError`, and a negative allocation would increase
availability on grant. *Rule:* an omitted resource is 0 and that is backward
compatible; a present value must be a non-negative integer; a claim on a
resource the device does not declare is rejected.

**5. Where strategy lives.** Some callers need several reservations granted
together or not at all (a base model and its LoRA on separate leases); others
want whichever of a set fits now (ten voices for a book render). Both tempt a
smarter daemon with partial grants. *Rule:* the daemon stays all-or-nothing
over whatever list it receives. The solver decides the list, and a plan is a
sequence of atomic requests: required groups together first, then one request
per optional group, so a lost race drops one voice rather than the book.

**6. The consequence of free names.** Once `kind` is authoritative, a device
name no longer has to be `cuda:GPU-<uuid>`, so the UUID moves into metadata.
A worker launched by `reserve` learns its grant only from the environment, so
it cannot look the UUID up. *Rule:* a grant is the request plus the device's
immutable metadata. The daemon is the authority for what a worker needs.

## The model

A device is exactly `name`, `resources`, and `metadata`.

**Resources** are non-negative integers. They are claimed on grant, returned
on release, and checked atomically across a request. The daemon attaches no
meaning to a resource name; `ram_mb`, `compute_units`, and `exclusive` are
conventions between the operator and callers. `exclusive: 1` gives mutual
exclusion among callers who claim it, with no new code.

**Metadata** values are scalars only: string, integer, float, or boolean.
`kind` is required on every device and is authoritative for mechanics: which
environment variables to set, how torch finds the device. A device of kind
`cuda` must carry `uuid: "GPU-<uuid>"`. Names are otherwise free, so the daemon
validates the UUID in metadata, not in the name. Everything else in metadata
is opaque to the daemon and exists for solvers: `fp16`, `nvfp4`, `nvlink`,
`model`, `power_pool`.

**Value rules** for a resource map, at config load, on the wire, and at the
CLI: an omitted resource is 0, so `{}` is a valid allocation that claims
nothing; a present value must be an integer that is not a boolean and is not
negative; `null`, negative, string, and fractional values are invalid; a claim
on a resource the device does not declare is rejected, because the daemon
cannot account for it.

**One namespace per device.** A device's resource names and metadata keys
must be disjoint, and neither may be `name`. The daemon checks this at config
load. The two maps exist so the daemon knows what to account and what to
report; a caller refers to a key as `a.ram_mb` or `a.kind` without saying
which map it came from, because the clause it appears in already says so
(see step two).

The daemon stays explicit and best effort. A solver reads one snapshot and
plans against it; the daemon re-checks on grant; a stale snapshot means a
rejection and a re-plan. No queuing is added.

## Shapes

Config:

```json
{ "devices": [
  { "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780",
    "resources": { "ram_mb": 24576, "compute_units": 84 },
    "metadata":  { "kind": "cuda",
                   "uuid": "GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780",
                   "model": "NVIDIA RTX A6000" } },
  { "name": "cpu", "resources": { "ram_mb": 65536 }, "metadata": { "kind": "cpu" } }
] }
```

Request allocation on the wire:

```json
{ "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780", "resources": { "ram_mb": 8192 } }
```

`resources` may be omitted. Any other key is malformed.

Granted allocation on the wire:

```json
{ "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780",
  "resources": { "ram_mb": 8192 },
  "metadata": { "kind": "cuda", "uuid": "GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780", "model": "NVIDIA RTX A6000" } }
```

The grant is the request plus the device's immutable metadata. A worker
launched by `reserve` learns its grant only from `RESERVED_GRANTED_ALLOCATIONS`,
so the grant must carry what the worker needs to build `CUDA_VISIBLE_DEVICES`
and find its torch device. The daemon is the authority for that, which is more
robust than joining a grant against a status snapshot the launcher happened to
fetch.

Status device on the wire:

```json
{ "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780",
  "capacity":  { "ram_mb": 24576, "compute_units": 84 },
  "available": { "ram_mb": 16384, "compute_units": 84 },
  "metadata":  { "kind": "cuda", "uuid": "GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780", "model": "NVIDIA RTX A6000" } }
```

Status reservation allocations are `{"name", "resources"}`.

These are format changes to the config, the request, the grant, and the
status message. The project is pre-1.0 and the compatibility policy allows
that; the docs already say to install the same version everywhere. This is
the one break worth taking once.

## Changes by layer

### `reserved/core.py`

- `Resources`: a frozen record over a sorted tuple of `(name, int)` pairs,
  with `from_mapping()` as the validating constructor and mapping-style
  access. A frozen dataclass does not freeze a dict, and the record must stay
  hashable inside other frozen records.
- `Metadata`: same shape; values must be `str`, `int`, `float`, or `bool`,
  never `None`; keys are non-empty strings.
- `Device(name, capacity: Resources, metadata: Metadata)` with a `kind`
  property. `DeviceAvailability(device, available: Resources)`.
- `DeviceAllocation(name, resources)` stays the request and ledger record.
  New `GrantedAllocation(name, resources, metadata)` is what a grant returns.
- `DeviceStatus(name, capacity, available, metadata)`.
- `load_ledger()`: each device must have a string `name`, an optional
  `resources` map, and a `metadata` map whose `kind` is a non-empty string.
  Any other key is an error. A `cuda` device must carry a valid `uuid`.
  Resource names and metadata keys must be disjoint and must not be `name`.
  Duplicate device names are an error; today they silently shadow in the
  index.
- `cuda_uuid(metadata)` replaces the name-based parser.
- `grant()`: per allocation, the device must exist; each claimed resource
  must be in `device.capacity`, else `unknown resource 'x' on device 'y'`;
  available must cover the claim, else `insufficient x on 'y': need n, have
  m`; subtract. The grant joins the device's metadata. `release()` adds back
  per resource. Neither function names a resource.

### `reserved/protocol.py`

- `allocation_from_wire()` for requests and `granted_allocation_from_wire()`
  for grants and `RESERVED_GRANTED_ALLOCATIONS`. Both reject unknown keys and
  delegate value validation to the core records.
- Serializers emit every key on every record; the status serializer emits the
  nested device shape.
- `daemon_state._handle_reserve` already maps `KeyError`, `TypeError`, and
  `ValueError` to a `malformed request` response with the ledger unchanged.

### `reserved/sdk/solvers.py` (interim, replaced in step two)

- `_matches`: `constraint.kind` against `metadata.kind`, the name pin, and
  each requested resource present in `available` with at least the amount.
- `_place`: the allocation's resources are the constraint's resources,
  subtracted from a working copy of `available`.
- Ranking without naming a resource: mean available fraction over the
  device's resources with nonzero capacity. Consolidate takes the lowest,
  Spread the highest. Existing solver tests, which rank by RAM, must pass
  with their expectations unchanged.

### `reserved/sdk/client.py`

- `DeviceConstraint(kind=None, *, name=None, **resources: int)`. The first
  positional argument keeps its meaning, so `DeviceConstraint("cuda",
  ram_mb=8192)` is unchanged. The parameter is renamed from `type` to `kind`,
  and resources are validated at construction.
- `Reservation.devices` becomes `list[GrantedAllocation]`.
  `.resources["ram_mb"]` and `.metadata["kind"]` replace `.ram_mb`,
  `.compute_units`, and prefix parsing. `GrantedAllocation` is exported.
- `torch_device(granted)` dispatches on `metadata["kind"]`; CUDA uses
  `metadata["uuid"]`.
- `_lease_environment`: `CUDA_VISIBLE_DEVICES` from the UUIDs of cuda-kind
  grants in first-grant order, distinct; `RESERVED_DEVICE` is the first grant
  whose kind is not `cpu`, else the first grant; `RESERVED_GRANTED_ALLOCATIONS`
  holds the granted records including metadata.

### `reserved/bin/reserve.py`

- `type=` and `name=` remain the constraint keys, so documented command lines
  keep working; the docs call `type` the device kind. `ram` and `compute`
  stay as aliases through a two-entry table. Every other key is a resource
  claim, validated through `Resources.from_mapping`, so `ram=-1` or
  `ram=abc` is a usage error naming the key.

### `reserved/bin/reserve_status.py` and the dashboard in `reserved/daemon.py`

- Human output per device: name, kind, `available: a/c <resource>` for each
  declared resource in sorted order, then remaining metadata as `k=v`.
  Allocation lines list claimed resources. The words `MB` and `compute`
  leave the output.
- HTML: Name, Kind, one column per resource in the sorted union across
  devices, Metadata. Reservations: ID, Label, Age, Device, one column per
  resource in the union across allocations. Missing cells show `—`.

### `reserved/bin/generate_config.py`

- Emits the new shape: name `cuda:GPU-<uuid>` as a default, `resources` with
  `ram_mb` and `compute_units`, `metadata` with `kind`, `uuid`, and `model`
  from torch's device name. Still validated through `core.load_ledger`.

### Config files, examples, docs

- Convert `reserved.example.json` (also fixing its stale `cuda:0` names and
  its `model` field, which becomes metadata), `examples/cpu.json`,
  `examples/cuda.json`, `examples/cuda_dual.json`, `examples/m2_mac.json`.
- `README.md`, `ARCHITECTURE.md`, `RESERVED_DESIGN.md`, `GETTING_STARTED.md`,
  and `examples/README.md`: config snippets, the three categories and value
  rules, wire examples, status samples, `GrantedAllocation` fields, and the
  CUDA section (the UUID lives in metadata and the mask is built from it).

## Tests

Fixtures are reshaped first across the suite, including `tests/cuda_support.py`
and the `daemon_env` fixture in `tests/test_inherited_lease.py`. The
inherited-lease suite is the real check that CUDA masks and torch mapping
work from metadata.

New or changed assertions:

- Core: `Resources` and `Metadata` validation tables (absent, 0, positive,
  `-1`, `None`, `True`, `"5"`, `2.5`, empty key; metadata rejects `None` and
  nested values); `load_ledger` rejects a device without `kind`, a cuda device
  without a valid `uuid`, an unknown top-level key, and a duplicate name;
  grant rejects an unknown resource with the ledger unchanged; grant and
  release on a device declaring only a custom resource `widgets`; the grant
  carries the device's metadata; exact fit and one-over cases.
- Protocol: request round trip; omitted `resources` parses to empty; unknown
  key, `null`, `-1`, `"5"`, and `true` raise; granted round trip includes
  metadata; nested status shape.
- Daemon state: a negative claim and an unknown resource each yield a failure
  response and leave the ledger and leases unchanged.
- Solvers: kind matching via metadata; a resource the device lacks does not
  match; existing ranking expectations hold; a device set with only
  `widgets` solves.
- CLI: aliases work; `foo=3` becomes a claim; `ram=-1` and `ram=abc` are
  usage errors; the exec environment builds the CUDA mask from metadata.
- SDK: `DeviceConstraint("cuda", ram_mb=-1)` raises; `torch_device` on a
  cuda grant under a reordered mask (existing parametrization); a grant of
  unknown kind raises; `child_env` to `adopt_inherited_lease` round trip
  preserves resources and metadata.
- Generator: records have the three categories; `uuid` equals the name's
  UUID; `model` is torch's device name.
- Real daemon: a config with `widgets: 2` grants two one-widget leases,
  rejects a third naming `widgets`, and grants it after a release. This
  proves a new resource needs no code.

## Verification

1. `make check` after the last code change; record the pytest summary line.
2. Manual on a CPU-only machine: daemon with the converted `examples/cpu.json`
   on a demo socket; `reserve-status`; `reserve demo ram=256 -- sleep 30`;
   `reserve demo ram=-1 -- true` (usage error); a raw socket request claiming
   `widgets` (rejected, status unchanged); the HTTP dashboard.
3. Real-card CUDA mapping on the 4x A6000 machine: each granted UUID resolves
   to the torch device with that UUID under the launcher's mask and in an
   unmasked process. If this is not run before merge, name it as a gap.
4. Every JSON example in the markdown files parses and loads with
   `core.load_ledger`.

## Observable changes to document

- Config format: three categories; `type` becomes `metadata.kind`; CUDA
  `uuid` in metadata.
- Request and grant allocation records nest `resources`; grants carry
  `metadata`; status devices nest `capacity`, `available`, and `metadata`.
- Human status wording; stricter rejection of malformed values; rejection of
  unknown resources.
- `GrantedAllocation` replaces `DeviceAllocation` in `Reservation.devices`;
  the `DeviceConstraint` parameter `type` is renamed `kind`.

## Steps two and three

**Step two: constraint language and solver.** A request may claim several
reservations; each reservation may claim several devices; reservations
release independently because callers optimize globally across applications.
Variables are scoped per reservation and referenced across reservations by
label, as `alpha.a`. Variables do not bind distinct devices by default; the
base model and its LoRA may share a card, and two cards are requested with
`c.name != d.name`. Two clauses exist, and the clause determines what an
operand means. In `where`, every operand is immutable: `var.name` is the
device's identity, `var.<key>` is a metadata scalar or a resource's
configured total, and literals are strings, integers, floats, or booleans;
operators are `==`, `!=`, `<`, `<=`, `>`, `>=`, with several constraints
forming an implicit AND. Remaining capacity never appears in a predicate.
In `consume`, `var.<key> = n` claims a resource. Because a device's resource
names and metadata keys are disjoint, no prefix is needed to tell them
apart; "a card with over 24 GB" is `where c.ram_mb >= 24576`, and "claim 10
GB of it" is `consume c.ram_mb = 10240`. The solver copies the snapshot's
remaining capacity, assigns devices to variables by backtracking, subtracts
every claim from every reservation as it goes, and rejects any assignment that
takes any resource on any device below zero. Consolidate and Spread become
candidate orderings. The output is today's explicit request. The CLI and SDK
share one string grammar; today's `DeviceConstraint("cuda", ram_mb=8192)`
desugars into it.

The sample that drove the design:

```text
request
  reservation alpha
    where   a.kind == "cuda"
            a.nvfp4 == true
    consume a.ram_mb = 10240
  reservation beta
    where   b.name == alpha.a.name
    consume b.ram_mb = 2048
  reservation delta
    where   c.kind == "cuda"
            d.kind == "cuda"
            c.name != d.name
    consume c.ram_mb = 10240
            d.ram_mb = 20480
```

**Grouping and optional reservations.** Some callers want several
reservations granted together or not at all (a base model and its LoRA held
on separate leases) while others want whichever of a set fits now and the
rest later (ten voices for a book render). The daemon does not learn about
either. It stays all-or-nothing over whatever list it receives, and the
solver decides what goes in that list. In the language, every reservation
belongs to a group; ungrouped reservations form one implicit required group,
so existing code behaves as today. A group may be marked `optional`. The
solver places required groups first, then optional groups in declared order
against its working copy, drops optional groups that do not fit, and emits
explicit requests for the placed set (one for all required groups, then one
per optional group; see "A plan is a sequence of atomic attempts" below).
The SDK result names the granted reservations and the dropped labels with a
reason each; a dropped optional reservation is a planning outcome, not an
error. Cross-reservation references may point within a group or from an
optional group to a required one, never from a required group into an
optional one, since a required reservation cannot depend on a device that may
never be chosen. If the daemon rejects the required request because the
snapshot went stale, the attempt fails and the caller re-plans, which is the
existing contract. Status shows granted optional reservations as ordinary
independent reservations, which they are.
"At least K of the optional groups" is a natural later extension.

```text
request
  group pair                                   # required by default
    reservation base  where a.kind == "cuda"          consume a.ram_mb = 8192
    reservation lora  where b.name == pair.base.a.name consume b.ram_mb = 2048
  reservation voice-1 optional  where v.kind == "cuda" consume v.ram_mb = 4096
  reservation voice-2 optional  where v.kind == "cuda" consume v.ram_mb = 4096
```

**Step three: the service feature.** A device of kind `service` with a
`concurrency` resource, for example `service:anthropic` at 8. A per-call lease
claims `concurrency: 1` and releases on completion; a worker-lifetime lease
claims its allowance at start. Mechanics for a kind with no environment mask,
documentation, and a real-daemon test bounding parallel calls. Expected to
touch no core, protocol, or solver code.

## Layering and the SDK shape

From the bottom up, each layer knows strictly less about the caller's intent.

1. **Daemon.** Ledger, check-and-subtract, lease fds. Speaks only explicit
   named allocations. All-or-nothing per request. Knows nothing about kinds
   beyond validating a CUDA UUID, nothing about groups, nothing about why.
2. **Protocol.** Record shapes and value validation. No policy.
3. **Transport.** Fetch a snapshot; send one request and receive a grant with
   its fds. One function each.
4. **Planner.** A pure function from a query and a snapshot to a plan. All
   strategy lives here: matching metadata, consuming a working copy, ordering
   groups, dropping optional groups, placement preference. No IO, so it is
   testable with dict fixtures and usable as a dry run.
5. **Acquisition.** Runs snapshot, plan, submit, and maps fds to
   `Reservation` handles. Owns the lease lifetime API: `check()`,
   `release()`, `popen()`.
6. **Mechanics.** `torch_device`, environment masks, lease handoff. Keyed on
   `metadata.kind`. The only layer that knows what a CUDA card is.
7. **CLI.** Parses one reservation into a query, calls layers 4 through 6,
   and execs.

The dividing line sits between 3 and 4. Everything below is subtraction;
everything above is proposal.

**A plan is a sequence of atomic attempts.** The planner emits an ordered
list of wire requests, each all-or-nothing at the daemon. All required
groups go in the first request together; the daemon already provides that
atomicity. Each optional group becomes its own request, in declared order.
The SDK submits them in sequence. If the required request is rejected,
nothing optional is attempted and the caller gets an error. If an optional
request is rejected because another client won a race, that group is
reported as dropped with the reason "rejected at grant", distinct from "did
not fit the snapshot". Races degrade one optional group at a time instead of
failing the whole set, at the cost of a few extra round trips on a local
socket.

**SDK shape, illustrative and for review only:**

```python
# Pure planning, no IO. Also a dry run: "how many voices fit right now?"
plan = request.plan(snapshot)
plan.attempts      # ordered atomic wire requests
plan.dropped       # label -> "did not fit"

# Acquisition
outcome = request.acquire_available()
outcome.granted    # label -> Reservation, each holding its own lease
outcome.dropped    # label -> reason: "did not fit" or "rejected at grant"
```

`Request.acquire()` keeps its current contract for all-required requests: a
tuple of reservations, or an exception. Only requests with optional groups
need the richer outcome, so existing code is unchanged. The dry run answers
the question `WORKER_GROWTH_PROPOSAL.md` set out to answer, as a count of
placed labels, with no separate helper.

**Patterns by workload:**

| Workload | Entry point | Expected pattern |
| --- | --- | --- |
| One training job | `with Reservation(...)` | Acquire or raise. The caller decides when to retry. |
| Checkpoint plus LoRA | `Request(...).acquire()` | Atomic acquire, release independently. Unchanged. |
| Ten voices, elastic pool | `Request(..., optional)` with `acquire_available()` | The loop below. |
| Orchestrator with its own placement | pinned names, or explicit requests | Enters at layer 3. Owns re-planning. |
| Non-Python worker | `reserve` CLI | Single reservation, lease by fd. Unchanged. |
| Service call (step three) | `with Reservation("call", ...)` per call, or one allowance per worker | Short lease in a loop, or one long lease. |

The elastic pool pattern developers should expect:

```python
pending = {f"voice-{i}": voice_spec for i in range(10)}
held = {}
while pending or held:
    if pending:
        outcome = Request(
            *[Reservation(label, *voice_spec, optional=True) for label in pending]
        ).acquire_available()
        for label, reservation in outcome.granted.items():
            held[label] = start_render(reservation)
            del pending[label]
    done = wait_for_any(held)          # the caller's own work loop
    for label in done:
        held.pop(label).release()      # capacity returns; the next pass re-plans
```

Dropped voices are normal, not errors. The caller re-plans when its own
releases return capacity, or after a backoff, rather than spinning. The
daemon never pushes notifications, and the design keeps it that way.

Deferred beyond these steps: a client-side bounded wait on a full service,
weighted tokens-in-flight claims, provider placement across local and hosted
services, API key distribution and rotation, and cumulative token budgets.
The last two change stated guarantees and need their own design documents.
