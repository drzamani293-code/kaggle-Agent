"""
Specialized prompts for the Stanford RNA 3D Folding Part 2 competition.

These prompts override the generic planner and code generation prompts with
domain-specific RNA structure prediction knowledge.
"""

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate


RNA_PLANNER_PROMPT = PromptTemplate.from_template(
    """\
You are a structural bioinformatics expert and Kaggle grandmaster competing in
the Stanford RNA 3D Folding Part 2 competition.

Your task is to create a winning plan for predicting RNA 3D structures from sequences.
The metric is TM-score (best-of-5 predictions, higher = better).

## Competition-Specific Knowledge

### What Won in Part 1
- **Protenix-v1** (ByteDance): AF3-surpassing, RNA MSA+template support → top performer
- **RhoFold+**: Best single-chain RNA predictor (Nature Methods 2024)
- **Boltz-1** (MIT): Open-source AF3 alternative, diverse diffusion predictions
- **Template-based modeling**: TM-score 0.9+ when homologs exist vs 0.4-0.6 de novo
- d4t4 team: Protenix (weight 0.75) + TBM (0.15) + Boltz (0.10) → select 5 from 15

### Key Technical Facts
- Only C1' atom coordinates needed (not full-atom)
- Submit 5 diverse predictions per RNA (scored on BEST-of-5)
- TM-score > 0.45 = correct global fold
- RNA up to 6,000 nucleotides in Part 2 (new challenge)
- Novel folds = no templates → must use de novo methods

### Proven Libraries/Tools
- `boltz` (pip): Boltz-1 predictions
- RhoFold+ (GitHub: ml4bio/RhoFold): best single-chain
- Protenix (GitHub: bytedance/Protenix): AF3-level
- `biopython`: PDB parsing, BLAST
- `nhmmer`/`blastn`: MSA generation and template search

## Problem Context

**Problem Description:**
<DESCRIPTION>
{problem_description}
</DESCRIPTION>

<ANALYSIS.QUANTITATIVE>
{quantitative_analysis}
</ANALYSIS.QUANTITATIVE>

<ANALYSIS.QUALITATIVE>
{qualitative_analysis}
</ANALYSIS.QUALITATIVE>

## Required Plan Structure

Generate a step-by-step plan that MUST include:
1. Setup and dependency installation (Protenix, RhoFold+, Boltz-1)
2. Load test_sequences.csv and preprocess sequences
3. MSA generation using nhmmer/BLAST against RNAcentral
4. Template search (BLAST against PDB RNA structures)
5. Multi-model prediction loop:
   a. Protenix-v1 (5 seeds per sequence)
   b. RhoFold+ (4 seeds per sequence)
   c. Boltz-1 (3 seeds per sequence)
   d. Template-based prediction (when available)
6. Greedy diverse ensemble selection (quality + diversity balance)
7. Format and write submission.csv (C1' coordinates for 5 predictions)
8. Validate submission format

The plan should handle the 9-hour compute budget on 2x T4 GPUs efficiently.
"""
)


RNA_CODE_GEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """\
You are a structural bioinformatics expert and expert Python developer competing in
the Stanford RNA 3D Folding Part 2 Kaggle competition.

## Competition Setup
- **Task**: Predict 3D C1' atom coordinates for RNA sequences
- **Metric**: TM-score, best-of-5 predictions (higher is better)
- **Input**: `./input/test_sequences.csv` with columns: target_id, sequence
- **Output**: `./output/submission.csv` with C1' coordinates for 5 predictions
- **GPU**: 2x NVIDIA T4, 16GB each, 9-hour time limit

## Submission Format
```
ID,resname,resid,x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3,x_4,y_4,z_4,x_5,y_5,z_5
{target_id}_{resid},{A/U/G/C},{1..L},{coords for pred 1},...,{coords for pred 5}
```

## Proven Model APIs

### Boltz-1 (installed via pip install boltz)
```python
# Write YAML, run CLI
yaml_content = f"version: 1\\nsequences:\\n  - rna:\\n      id: A\\n      sequence: {{seq}}"
# boltz predict input.yaml --out_dir output/ --seed 42 --diffusion_samples 1 --accelerator gpu
```

### RhoFold+ (GitHub: ml4bio/RhoFold)
```python
# python inference.py --input_fas query.fasta --output_dir output/ --device cuda:0 --ckpt weights.pt
# Output: relaxed_1000_model.pdb with C1' atoms
```

### Protenix-v1 (GitHub: bytedance/Protenix)
```python
# Input JSON format: {{"name": id, "sequences": [{{"type": "rna", "id": "A", "sequence": seq}}], "modelSeeds": [0,1,2,3,4]}}
# python runner/inference.py --input_json_path input.json --output_dir output/ --device cuda:0
```

### Parse C1' coordinates from PDB
```python
def parse_c1prime(pdb_path):
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                atom = line[12:16].strip()
                resname = line[17:20].strip()
                if atom == "C1'" and resname in ("A","U","G","C"):
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(coords)
```

### Ensemble Selection
```python
def kabsch_align(mobile, target):
    mob = mobile - mobile.mean(0); tgt = target - target.mean(0)
    U, S, Vt = np.linalg.svd(mob.T @ tgt)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1,1,d]) @ U.T
    return mob @ R.T + target.mean(0)

def fast_tm_score(pred, ref, n=50):
    L = len(pred); d0 = max(0.6*(max(L,15)-0.5)**(1/3)-2.5, 0.5)
    idx = np.linspace(0,L-1,min(n,L),dtype=int)
    p = kabsch_align(pred[idx], ref[idx])
    d = np.sqrt(((p - ref[idx])**2).sum(1))
    return float(np.mean(1/(1+(d/d0)**2)))
```

## Problem Description
{{problem_description}}

## Evaluation Metric
{{evaluation_metric}}

## Plan
<Plan>
{{plan}}
</Plan>

## Available Libraries
{{pkg_str}}
""",
        ),
        MessagesPlaceholder("history"),
        (
            "user",
            """\
Please write code for this task. The code must run in a Jupyter notebook cell.
Do NOT include visualizations or plots.
Ensure all file paths use `./input/` for reading and `./output/` for writing.

<CurrentTask>
{current_task}
</CurrentTask>
""",
        ),
    ]
)
