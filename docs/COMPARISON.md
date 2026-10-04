# Matched flow and BTM experiment

The first experiment uses PushT and the frozen LeWM encoder. The baseline is this
fork's matched 16-step flow planner; the method is the one-step BTM planner. This
does not reproduce the published LeFlow architecture exactly and does not yet
implement V-JEPA 2.1/Meta-World.

## Launch on the GPU server

Complete the Conda, data download, latent caching, and W&B login steps in
[TRAINING.md](TRAINING.md) first. Check `nvidia-smi` and set
`CUDA_VISIBLE_DEVICES` to the GPUs allocated to this experiment. The launcher
does not reserve GPUs through a cluster scheduler or terminate other jobs.

```bash
conda activate btm-jepa
export STABLEWM_HOME=/data/leflow
export CUDA_VISIBLE_DEVICES=0,1,2,3  # replace with the allocated GPU indices

# Inspect the fully resolved configuration before starting.
python scripts/run_comparison.py --name pusht_pair_v1 --gpus 4 --dry-run

# A detached foreground supervisor runs flow and then BTM on the same allocation.
# Each method has a separate log, checkpoint directory, and online W&B run.
nohup python -u scripts/run_comparison.py --name pusht_pair_v1 --gpus 4 \
  > "$STABLEWM_HOME/pusht_pair_v1.launch.log" 2>&1 < /dev/null &
```

Use a scheduler instead of `nohup` if the machine requires one. Only one method
runs at a time, using no more than four GPUs; intermediate evaluation uses a
training GPU. This also avoids baseline/method contention during timing. The
campaign has a process lock and stops on any training or online W&B failure.

The launcher freezes the git revision, resolved training configurations, cache
manifest hash, encoder hash, and exact episode split in
`$STABLEWM_HOME/runs/pusht_pair_v1/comparison.json`. It refuses dirty code, fixture
data, changed configuration on resume, or a changed data manifest between jobs.
Both methods use the same global batch, epochs, data ordering seed, architecture
dimensions, inverse loss, short-step dynamics consistency loss, and controller
budget. FM adds its time-conditioning parameters; BTM uses its stationary-map
objective, so their raw generative losses are not directly comparable.

Common overrides apply to both methods, for example
`epochs=10 evaluation.episodes=20`. On three GPUs use
`global_batch_size=96 micro_batch_size=32` for **both** methods. For a stopped
campaign, rerun the identical command with `--resume`; finished jobs are skipped
and unfinished jobs resume their last checkpoint. Do not change git revisions
mid-campaign; start a new campaign for a method change.

## Measurements and development targets

| Priority | Measurement | Development target |
|---|---|---|
| Primary | Actual task success at goal offset 100, fixed 200-action budget | Improve long-horizon validation success relative to matched FM |
| Efficiency | Full-controller latency, mean and p95, on the same GPU | Preserve success while reducing latency; 1 vs 16 generator calls alone is insufficient |
| Coverage | Success at offsets 25 and 50; mean across 25/50/100 | Check that a long-horizon gain does not hide a short-horizon regression |
| Diagnostics | Observed subgoal latent error after an executed chunk, predicted rollout goal error, sample diversity | Identify whether failure comes from proposals or execution; these are not physical reachability certificates |
| Cost | World-model state predictions, GPU memory, training throughput | Record the cost of any improvement |

W&B groups both runs under `pusht_pair_v1` in project `btm-jepa`. Plot
`eval/offset_100/success_rate`, `eval/mean_success_rate`,
`eval/offset_100/planning_batch_latency_ms_mean`,
`eval/offset_100/planning_batch_latency_ms_p95`, and
`eval/offset_100/observed_subgoal_mse_after_chunk` against optimizer step.
`best.pt` is selected by mean validation success across the three offsets for
both methods. The separate offset-100 result is the primary research endpoint.

One seed is a pilot. Use it to identify failures and develop on **validation**.
For confirmation, use a new campaign with at least three matched training seeds:

```bash
python scripts/run_comparison.py --name pusht_pair_confirm --gpus 4 \
  --seeds 3072 3073 3074
```

No numeric success target is asserted before measuring the baseline. Matching
success with lower total controller latency is the first useful result;
repeatable improvement at offset 100 is the stronger research target.

## Final held-out comparison

Freeze the method before using the test set. After both training jobs finish,
evaluate their validation-selected checkpoints on identical test episodes and
start/goal pairs. Run sequentially on the same otherwise-idle GPU. The example
below uses the 3072 seed; repeat for each confirmation seed.

```bash
campaign="$STABLEWM_HOME/runs/pusht_pair_v1"
for offset in 25 50 100; do
  for method in flow btm; do
    CUDA_VISIBLE_DEVICES=0 python eval_subgoals.py \
      --checkpoint "$campaign/${method}_3072/best.pt" \
      --split test --episodes 50 --seed 42 --goal-offset "$offset" \
      --budget 200 --mode hierarchical --spacing 2 --execute-blocks 1 \
      --candidates 64 --flow-steps 16 \
      --cem-candidates 32 --cem-iterations 3 --cem-elites 8 \
      --output "$campaign/test/${method}_3072_offset_${offset}.json"
  done
  python scripts/compare_evaluations.py \
    --flow "$campaign/test/flow_3072_offset_${offset}.json" \
    --btm "$campaign/test/btm_3072_offset_${offset}.json" \
    --output "$campaign/test/comparison_3072_offset_${offset}.json"
done
```

Replace the evaluation GPU index with an allocated GPU. Both splits must have
enough distinct eligible episodes; evaluation fails rather than silently sampling
with replacement. If fewer than 50 test episodes support an offset, choose a
smaller common count **before** reading outcomes. Evaluation records each
episode/start index, controller settings, training protocol, encoder and manifest
hash, hardware, checkpoint hash, and individual success outcome.

The comparison script rejects unmatched protocols and fixture results. It reports
BTM-minus-flow success in percentage points, a paired episode bootstrap interval,
discordant outcomes, and measured controller speedup. The interval is conditional
on a single trained model pair; it is not training-seed uncertainty or proof of
equivalence. Report per-seed results and mean/standard deviation across training
seeds. Hardware/load can still differ despite matching protocol metadata, so
inspect the recorded hardware and run timings before claiming a speedup.

Intermediate validation is online in W&B; standalone final test reports are
saved as JSON by the commands above. No actual GPU campaign has been launched
from the development workspace: `target_server_2` requires its working SSH
configuration, jump-host tunnel, and authorized key/agent in the execution
environment.
