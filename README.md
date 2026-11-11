# DHD-SDDP
Extension of SDDP.jl to decision-hazard-decision formulation

## Origin

This repository is a fork of [SDDP.jl](https://github.com/odow/SDDP.jl),
imported at commit
[`14d9d805`](https://github.com/odow/SDDP.jl/commit/14d9d805) (v1.15.0) and
distributed under the same MPL-2.0 license (see `LICENSE.md`). The package keeps
the name `SDDP` and its UUID, so code written for SDDP.jl runs unchanged.
Nodes without here-and-now variables follow the original hazard-decision code
path.

To see every change made on top of SDDP.jl, diff against the commit that
imported it.

## Layout

```
src/            SDDP.jl source; DHD-specific code lives in src/dhd/
test/           SDDP.jl tests (unchanged, used as a regression suite);
                DHD tests live in test/dhd/, one file per src/dhd/ file
examples/       HD and augmented-DH toy models used to validate the DHD solver
```

## Usage

The `test/` and `examples/` environments load this fork through
`[sources] SDDP = {path = ".."}` (requires Julia 1.11 or later). Outside these
environments, `using SDDP` loads the registered package.

Run the full test suite:

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

Run a single test file:

```
julia --project=test test/dhd/expansion.jl
```

Run an example:

```
julia --project=examples examples/hydro_dh_aug.jl
```
