# Reserved review incidents

## 2026-11-19: solver boundary obscured by representation changes

[PR #13 review](https://github.com/andrewguy9/Reserved/pull/13#pullrequestreview-5982844836)
identified ambiguous nested slot tuples, unchecked result cardinality,
exception-based no-placement handling, and batch policy inferred from its first
reservation. Renaming a result or changing argument order did not establish the
domain contract. Future solver changes must distinguish specifications, concrete
batches, alternative candidates, and leases before implementation, and state
whether every specification is required and how ordinary failure is returned.
