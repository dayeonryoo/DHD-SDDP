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

## Decision-hazard-decision nodes

Tag a control that must be chosen before the noise of its node is observed:

```julia
@variable(sp, 0 <= u_thermal <= 9, SDDP.HereAndNow)
```

A node with here-and-now variables becomes a decision-hazard-decision (DHD)
node: its here-and-now controls are chosen against every noise of the node, and
its other controls after the noise is observed. The cost-to-go functions stay
in the space of the original state variables.

Supported for DHD nodes:

- `SDDP.train` with the expectation, single cuts, `DefaultForwardPass`,
  `CompleteSampler`, `ContinuousConicDuality` and `Serial`
- `SDDP.simulate`, `SDDP.calculate_bound`, `SDDP.DecisionRule` and
  `SDDP.evaluate`
- `SDDP.deterministic_equivalent`
- writing, reading and re-adding cuts

Markovian policy graphs are supported and validated on finite horizons. In a
DHD node, the Markov state is known before the here-and-now controls are
chosen, and the transition to the next Markov state happens after the other
controls are chosen. The augmented-DH formulation commits one stage ahead,
before the next Markov state is known, so it is not equivalent on these
graphs.

The following raise an error for DHD nodes: other risk measures, multi-cut,
other forward passes, backward samplers, duality handlers and parallel schemes,
integer variables, objective and belief states, `SDDP.write_to_file` and
`SDDP.Inner.inner_dp`.

## Layout

```
src/            SDDP.jl source; DHD-specific code lives in src/dhd/
test/           SDDP.jl tests (unchanged, used as a regression suite);
                DHD tests live in test/dhd/: one file per src/dhd/ file, and
                validation.jl, which checks the examples against their
                augmented-DH or split HD equivalents
examples/       hydro and battery models in HD, augmented-DH and DHD form,
                a Markovian hydro model in DHD and split HD form, and a script that compares DHD and augmented-DH training
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
julia --project=test test/dhd/algorithm.jl
```

Compare DHD and augmented-DH training on the examples:

```
julia --project=examples examples/compare_dhd_dh_aug.jl
```
