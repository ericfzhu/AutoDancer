# Conformance traces

Put committed black-box traces in this directory. Use JSON Lines. The first line
is a header. Each later line is one action and the matching live state.

Every header must set `game_version` and `steam_build`. A trace can set
`ignored_paths` only for fields that the related rule does not test.

