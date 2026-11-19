# Reserved constraint language proposal

## Phase one

Use one ordinary device constraint record. Four fields are sufficient for the
minimum language:

| Field | Meaning |
| --- | --- |
| `type` | Optional exact device type, such as `cpu` or `cuda`. |
| `name` | Optional exact physical device name, such as `cuda:2`. |
| `ram_mb` | Exact memory consumption; defaults to zero. |
| `compute_units` | Exact consumption of the configured compute budget; defaults to zero. |

An omitted type or name imposes no restriction. A name is a configured physical
identifier, not a reservation label or a symbolic variable. Specifying both
type and name requires both to match.

The current SDK also supports `model=`. Retaining that existing compatibility
requires no new language machinery; this phase adds no model operators.

Examples use the existing constructor style, without proposing an expression
syntax:

```python
DeviceConstraint("cuda", ram_mb=10000, compute_units=50)
DeviceConstraint(name="cuda:2", ram_mb=10000, compute_units=50)
```

The first lets the solver choose a CUDA device. The second fixes the physical
device, including when describing an allocation whose placement is already
known. Converting a `DeviceAllocation` into such a constraint preserves its
name and amounts. Keeping an existing lease or acquiring another remains a
separate application action.

## Meaning of a requirement

One requirement describes one resource claim on one device. Its fields have
an implicit AND. RAM and compute are exact nonnegative integer consumption
amounts; the solver chooses placement, not quantities.

Matching checks the supplied device filters and whether the remaining capacity
can accommodate the claim. A successful selection produces:

```text
DeviceAllocation(selected_device.name, requirement.ram_mb, requirement.compute_units)
```

The requirement's amounts are copied unchanged. There is no memory interval,
minimum-consumption operator, or choice of how much to reserve.

A flat list requires every claim to be placed. Claims are additive: two entries
requesting 10000 MB and 5000 MB on the same device consume 15000 MB. Different
entries may share a device when capacity permits; a list does not imply distinct
devices. Named claims targeting the same device must fit together.

The inequalities belong to feasibility checking:

```text
sum of RAM claims on a device <= its available RAM
sum of compute claims on a device <= its available compute
```

Solving returns one complete candidate or an ordinary no-placement outcome.
It reads no daemon state and leaves its supplied inputs unchanged. An empty
list is a successful empty set of claims.

## Structure that permits later extensions

Keep the distinction between these two values:

- **Device constraint:** a description the solver interprets.
- **Device allocation:** a physical device name and exact amounts that
  acquisition and the daemon can use directly.

Phase one needs the constraint record and the existing pure placement
functions. The language is not a daemon protocol and need not become a
framework of expression classes in this patch.

Later, the same device constraint can be a leaf inside a richer description:

```text
OR(
    DeviceConstraint(type="cuda", ram_mb=10000),
    DeviceConstraint(type="cpu", ram_mb=10000)
)
```

That is a future input-model extension. Existing leaf requirements keep their
meaning. The solver input types can broaden when those expressions are added;
we are not promising that today's annotations or solver functions already
support every future expression.

Keep all interpretation in the pure solver. Callers and acquisition should
not inspect constraint variants or implement matching policy. Successful
results remain concrete allocations grouped into the application's reservation
lifetimes. The daemon continues to recheck capacity and atomically grant or
reject explicit requests.

This separation permits richer inputs without changing the allocation record,
wire format, lease mechanics, or resource accounting.

## Later extension catalog

These are possible additions, not phase-one implementation requirements.

| Extension | Meaning to settle when needed |
| --- | --- |
| More device predicates | Additional model operators, membership in a set, and other hardware properties. |
| Regex matching | Match device names by a pattern while still selecting concrete devices. |
| AND | Compose predicates on one device or require several additive claims; keep those scopes explicit. |
| OR | Offer alternative predicates or whole resource bundles, without implying a preference. |
| NOT | Negate predicates or device relationships; general negation of resource claims needs separate semantics. |
| Symbolic device bindings | Name a solver-selected device `A` so other claims can refer to it. |
| Same or different devices | Relate bindings without requiring the caller to know physical names. |

Memory-consumption ranges are unnecessary for the current goal. Future numeric
predicates on hardware properties would be separate from declaring consumption.
Choosing among feasible plans, enumerating alternatives, and selecting a subset
of reservations are also later solver-policy decisions.

## Bounded implementation

First establish the single constraint type and optional physical name. Then
adapt the pure solvers to that type and simplify their callers. Preserve
reservation grouping and the concrete acquisition interface; add no combinators
or generic extension machinery yet.

Verification should cover type and name matching, exact amounts, zero
consumption, insufficient capacity, and additive claims sharing a device.
Also check unchanged inputs and conversion from an allocation to an exact
constraint. Existing protocol and lease tests establish that the concrete side
retains its behavior.

This is a design proposal. The [current architecture](ARCHITECTURE.md#two-languages-explicit-and-constraint)
describes the implemented boundary; this document does not add SDK behavior.
