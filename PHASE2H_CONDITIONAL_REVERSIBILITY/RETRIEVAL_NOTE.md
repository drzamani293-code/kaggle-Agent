# Retrieval note — Phase 2H

**No primary source was readable during Phase 2H.** This is the same condition
recorded in Phase 2B (`LITERATURE_GAP_MAP.md`) and Phase 2G
(`PHASE2G_LITERATURE_COMPARISON/RETRIEVAL_LOG.md`), re-tested at the start and
again at the end of this phase.

## What was attempted, this phase

| target | URL | result |
|---|---|---|
| Kopra, arXiv:2202.13809 (abstract) | `https://arxiv.org/abs/2202.13809` | HTTP **403** |
| Kopra, arXiv mirror | `https://export.arxiv.org/abs/2202.13809` | HTTP **403** |
| Kopra, PDF | `https://www.arxiv.org/pdf/2202.13809v3` | HTTP **403** |
| Rowland, author page | `https://ericrowland.github.io/papers.html` | HTTP **403** |
| Jen, Physica D 45 (1990) 3–18 | `https://www.sciencedirect.com/science/article/pii/016727899090169P` | HTTP **403** |

The failure is at the proxy `CONNECT` stage, not at the origin. Verbatim from
`curl -v`:

```
> CONNECT arxiv.org:443 HTTP/1.1
< HTTP/1.1 403 Forbidden
* CONNECT tunnel failed, response 403
```

and from the proxy's own status endpoint:

```
"kind": "connect_rejected",
"detail": "gateway answered 403 to CONNECT (policy denial or upstream failure)",
"host": "arxiv.org:443"
```

The markdown-fetching tool returns the same 403. The block is host-independent
and applies to every external host tried in every phase, so it is an
environment policy, not a property of these publishers.

## Consequences for Phase 2H, stated plainly

1. **Section 1 does not reconstruct Rowland's reversibility theorem.** It
   derives conditional backward determinism from the local rule. Any agreement
   with Rowland's actual statement is unverified.
2. **Section 7 could not be done at all.** `KOPRA_PROOF_ADAPTATION.md` reports
   the task as **NOT COMPLETED**. Theorem 3.5 was never read, so "identify the
   exact step where width `w = 2` is used" cannot be answered about Kopra's
   proof. That document instead answers the question about *our own* proofs,
   and labels the substitution explicitly.
3. **Section 4's `2^n` mechanism is not attributed as a reconstruction.** The
   power-of-two structure reported there was found by measurement in this
   phase. Secondary summaries attribute a `2^n` nested-restart mechanism to
   Rowland; whether what we measured is that mechanism is **unknown**.
4. **No novelty claim is made anywhere in Phase 2H**, per the standing rule.

## Standing text

> **Novelty unresolved.** No result in this phase is claimed to be new, and
> none is claimed to be known. Where a source is named, the statement
> attributed to it is second-hand and is never used as a premise in a proof.
