# Native LeFlow self-refinement

The implementation previously delivered in `LeFlow-self-refinement-UNPUSHED.zip`
is now published in this repository. Source, tests, and configurations were
restored without changing their contents; existing concurrent repository work
was preserved.

See [method and run instructions](docs/SELF_REFINEMENT_IMPLEMENTATION.md).
The retry validation record is [SELF_REFINE_PUBLICATION.json](SELF_REFINE_PUBLICATION.json).

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/self_refine
```

The publication retry passed all 83 CPU tests. These are synthetic correctness
tests, not task-success measurements. No GPU training or native benchmark
evaluation was launched by the publication retry. The method is training-free
and requires existing native LeFlow/LeWM checkpoints for the real comparison.

`self_refine/` is an additive archived implementation. The separate existing
`leflow_refinement/` and `crossbench/` implementations remain unchanged.
