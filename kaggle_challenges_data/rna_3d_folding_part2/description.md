# Stanford RNA 3D Folding Part 2

## Competition Overview

Predict the 3D structure of RNA molecules from their sequences alone. This is one of biology's remaining grand challenges — unlike protein folding (solved by AlphaFold2), RNA 3D prediction remains difficult due to:
- Sparse experimental data (PDB has ~18,000 RNA structures vs ~200,000 protein)
- RNA's extreme structural flexibility and dynamic nature
- Presence of pseudoknots and complex tertiary interactions
- RNA complexes with proteins, small molecules (up to 6,000 nt in Part 2)

## Problem Statement

Given an RNA sequence (composed of nucleotides A, U, G, C), predict:
- The 3D coordinates (x, y, z) of the **C1' atom** of each nucleotide
- Submit **5 diverse structure predictions** per RNA sequence
- Scored by TM-score (best-of-5) against experimentally determined structures

## Key Challenges in Part 2

Part 2 introduces **harder targets** than Part 1:
- RNA molecules with NO available structural templates (novel folds)
- RNA complexes with small molecules and proteins
- Assemblies up to **6,000 nucleotides** (much larger than typical)
- cryo-EM structures of new-to-Nature RNA folds

## Organizers

- Stanford Medicine (Das Lab)
- HHMI Janelia Research Campus

## Prize Pool

$75,000 total:
- 1st place: $50,000
- 2nd place: $15,000
- 3rd place: $10,000

Plus: Top teams invited to co-author a peer-reviewed scientific paper.

## Timeline

- Entry deadline: March 18, 2026
- Final submission deadline: March 25, 2026

## Hosts

Rhiju Das Lab, Stanford Biochemistry
