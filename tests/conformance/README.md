# Conformance traces

Use `autodancer-record-trace` to collect version-pinned raw live evidence. The
recorder writes trace schema 2, stores each action with its matching live record,
and refuses to overwrite an existing file unless `--overwrite` is supplied.

Raw live observations are evidence, not automatic simulator assertions. Curate
small traces around one mechanic and add partial `observation`, `state`,
`events`, or `reward` expectations only for fields whose semantics the trace is
meant to establish. Use `ignored_paths` narrowly.

Legacy schema-1 state traces remain readable by `autodancer-compare-trace`.
