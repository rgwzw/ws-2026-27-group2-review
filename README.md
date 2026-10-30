# Quality-Aware Synthetic Data for Road Scene Understanding

This repository is the shared software framework for the seminar **Road Scene Understanding for Visually Impaired**. It studies whether synthetic-image quality, semantic consistency, structural preservation, diversity, and subset selection improve semantic segmentation on real SENSATION-DS scenes.

## Ground rules

- `datasets/sensation_ds_v2_1_1152x768/{training,validation,testing}` is authoritative and read-only.
- Synthetic data under `datasets/synthetic-data/` may augment training only.
- Validation and testing always contain real images; testing is reserved for final evaluation.
- A filename match is not enough for supervision. Synthetic records require verified image/mask geometry before segmentation training.
- Groups share one DeepLabV3Plus benchmark and result schema. Cross-group inputs are optional.

## Quick start

Use Python 3.11 or newer and install the project in a virtual environment:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
pytest
```

Build the synthetic index and the fixed instructor comparison subset:

```bash
python -m sensation_seminar.cli.build_manifest --data-root datasets --output synthetic-data-seminar/manifest.jsonl
python -m sensation_seminar.cli.make_subset --manifest synthetic-data-seminar/manifest.jsonl --method random --budget 0.25x --seed 42
```

Run the mandatory baselines for seeds 17, 42, and 73:

```bash
python -m sensation_seminar.cli.train --experiment real_only --seed 17
python -m sensation_seminar.cli.train --experiment random_budget_0.25x --seed 17
```

Repeat with seeds `42` and `73`. The secondary `random_budget_0.50x` baseline is supported but not required before the seminar.

## Repository map

- `architecture.md`: scientific and software design.
- `configs/`: immutable data definitions and reproducible experiment settings.
- `src/sensation_seminar/`: shared indexing, datasets, model, training, metrics, evaluation, and result code.
- `synthetic-data-seminar/`: manifest schema and generated metadata workspace; images are not copied.
- `baseline/`: instructor workflow and result location.
- `group-1/` ... `group-9/`: independent student tasks, code, tests, and result contracts.
- `tests/`: CPU-fast shared tests. Optional checks use `data`, `gpu`, `slow`, and `integration` markers.

The normal test suite neither downloads model weights nor modifies `datasets/`. See `baseline/README.md` for the full instructor workflow and each group README for its independent task and fallback behavior.

## Group overview

| Group | Students |
| --- | --- |
| 1 | Pritam Chaurasiya<br>Dhanjit Raj Gupta |
| 2 | Yingchao Li<br>Jiaying Ping |
| 3 | Mitesh Kumar |
| 4 | Jakub Planka<br>Abhiram Sharma Kommunuri Venkat Malikarjun Siva Sai |
| 5 | Aakash Girish<br>Kavitha Ganapathi Mahadeva Venkatesan |
| 6 | Khanh Linh Vuong<br>Keerthika Sathish |
| 7 | Revathi Bandi<br>Ritik Ritik |
| 8 | Bhaskar Sesha Sai Lingamaneni<br>Pranathi Rokkam |
| 9 | Indranil Paul<br>Keshav Periwal |


## Baseline Results

Before the seminar projects start, we provide two common segmentation baselines. These results serve as the reference point for all group-specific experiments.

All groups must use the same model architecture, dataset split, training configuration, and evaluation metrics. The main variable that should change between experiments is the selection of synthetic training images.

### Common Model

The common segmentation model is:

- **Model:** DeepLabV3+
- **Encoder:** MobileNetV3 Large 0.75
- **Classes:** 9 SENSATION-DS classes
- **Training data:** predefined SENSATION-DS training split
- **Validation data:** predefined real SENSATION-DS validation split
- **Test data:** predefined real SENSATION-DS test split

The validation and test splits must never be used for synthetic-image selection.

---

### Baseline Experiments

We provide two reference experiments.

#### Real Only

The first baseline uses only the original SENSATION-DS training images.

```text
real_only
```

#### Real + Random Synthetic Data

The second baseline adds a randomly selected synthetic subset.

```text
random_budget_0.25x
```

The synthetic budget corresponds to:

```text
0.25 × size of the real training set
```

With 2,190 real training images, this results in:

```text
548 synthetic images
```

The random subset is fixed and must remain unchanged for all group comparisons.

---

### Baseline Performance

| Experiment | mIoU | Road IoU | Sidewalk IoU | Road → Sidewalk Error |
|---|---:|---:|---:|---:|
| `real_only` | **0.6565 ± 0.0038** | **0.7053 ± 0.0073** | **0.8443 ± 0.0021** | 0.1190 ± 0.0020 |
| `random_budget_0.25x` | 0.6481 ± 0.0055 | 0.7002 ± 0.0057 | 0.8396 ± 0.0004 | **0.1140 ± 0.0051** |

The current results show that simply adding a random synthetic subset does **not automatically improve overall segmentation performance**.

Compared with `real_only`, the random synthetic baseline changes the results approximately as follows:

```text
mIoU:                  -0.0083
Road IoU:              -0.0051
Sidewalk IoU:          -0.0048
Road → Sidewalk Error: -0.0050
```

This means that the random synthetic subset slightly reduces overall segmentation accuracy, while at the same time producing a slightly lower Road-to-Sidewalk error rate.

This is an important motivation for the seminar:

> The goal is not simply to add more synthetic images.  
> The goal is to identify which synthetic images are actually useful for training.

---

## What Each Group Should Compare Against

Groups that create a synthetic-data selection strategy should compare their method against the fixed random baseline using the same synthetic budget.

The main comparison is therefore:

```text
real_only

vs.

random_budget_0.25x

vs.

group_method_budget_0.25x
```

The group-specific subset must also contain:

```text
548 synthetic images
```

This ensures that improvements are caused by the selection strategy and not simply by using more training data.

---

## Expected Group Experiment

For example, a quality-based group may produce:

```text
quality_budget_0.25x
```

A diversity-based group may produce:

```text
diversity_budget_0.25x
```

A semantic-consistency group may produce:

```text
semantic_consistency_budget_0.25x
```

The corresponding result should then be compared directly with:

```text
random_budget_0.25x
```

using the same model and training configuration.

---

## Main Question for the Seminar

The common question for all groups is:

> Can a more informed selection of synthetic images outperform a random synthetic subset of the same size?

A successful group method does not necessarily have to improve every metric.

For example, one method may improve overall mIoU, while another method may mainly improve safety-related metrics such as the Road-to-Sidewalk Error Rate.

Both effects are relevant for the overall seminar topic.

---

## Important Rules

Please do not change the common baseline setup unless this is explicitly part of your assigned task.

In particular, keep the following fixed:

- SENSATION-DS train/validation/test split
- segmentation architecture
- encoder
- input configuration
- optimizer and training settings
- evaluation metrics
- synthetic budget
- experiment seeds

The main experimental variable should be your method for evaluating or selecting synthetic images.

---

## About the Baseline Seeds

The baseline experiments use the common seeds:

```text
17
42
73
```

Group-specific downstream experiments should use the same seeds whenever possible.

Results should be reported as:

```text
mean ± standard deviation
```

across the common seeds.

---

## Interpretation

The current baseline already demonstrates an important point:

> More synthetic data is not automatically better.

The random synthetic baseline currently performs slightly worse than `real_only` in mIoU, Road IoU, and Sidewalk IoU.

At the same time, the Road-to-Sidewalk Error Rate is slightly lower.

The seminar therefore investigates whether quality-aware, consistency-aware, or diversity-aware selection can produce a synthetic subset that is more useful than random selection.

This baseline is the common reference for all seminar groups.