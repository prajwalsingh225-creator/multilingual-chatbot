# OOD eval set provenance

The eval set is the only instrument used to decide whether the `out_of_scope` class is
allowed to replace the rule-based fallback. Anything that moves its numbers moves a
promotion decision, so its history is recorded here.

## Frozen bytes

```
file:    backend/data/eval/ood_eval.json
sha256:  2e7d577145952b5dd85a661532e7572bb1ad9c582be9ec4500551f87293faf21
rows:    162 ood + 89 in_domain control
version: 1.1
```

The hash covers the file's exact bytes, so it also pins the `version` field and `notes`
inside it.

`tests/test_ood.py::test_the_eval_set_matches_its_recorded_hash` fails if the file's bytes
change without this hash being updated here and in `EVAL_SHA256` in that test. To make a
deliberate change: edit the file, then update both places in the same commit, and say why
in this file.

## Rules of use

- Never used for training, threshold tuning, or model selection.
- Never edited after seeing a candidate model's results. Every measurement below was taken
  on the bytes hashed above.
- `measure_ood.py` stamps `eval_file_sha256` into every report, so a report can never be
  silently read against different eval bytes.
- `--extra-set` is the only supported way to score new sentences, and it is measurement
  only.

## Change log

### 1.1 — 2026-10-01, before any candidate was scored

Three corrections. None of these were made after looking at a candidate's scores; each was
found by auditing the eval set against the training data.

**Two `other_domain` rows were actually in-domain track_order requests.** They were
token-permutations or accent variants of real training examples, so any model correctly
routing them to `track_order` was being counted as a *failure to reject*:

| language | was | actually is | problem |
| --- | --- | --- | --- |
| es | `dónde está mi paquete` | `donde esta mi paquete` | track_order es example 12, accent-only variant |
| hi | `मेरा सामान कहाँ है` | `सामान कहाँ है मेरा` | track_order hi example 30, word order reversed |

Replaced with genuinely off-topic sentences of the same language and kind
(`dónde está la sucursal del banco más cercana`, `मेरा सामान कहाँ रखा है`).

`normalize_text` casefolds and NFKC-normalises but does **not** strip diacritics, so
`test_the_eval_set_never_overlaps_training_data` could not see these. The audit compares
accent-stripped text and token multisets. `test_the_eval_set_shares_no_token_multiset_with_training`
now enforces that permanently. Nine gibberish/emoji rows (`???`, `@@@@###$$$`, `🙏`, ...)
legitimately share an empty token multiset and are exempted by requiring real word tokens.

**+26 Hinglish control rows** (63 → 89 `in_domain`). The control set had zero Hinglish
rows, so "false rejection" was only ever measured in en/hi/es while Hinglish was 25 of the
162 OOD rows. A model could reject Hinglish in-domain traffic at 100% and still pass.
The gate is now enforced on Hinglish separately
(`test_hinglish_control_requests_are_not_rejected`). Rows were written independently of
`intents.json` and `out_of_scope.json` and verified to collide with neither, exactly or
above 0.7 token Jaccard.

### 1.0 — 2026-09-30, commit eb7e5d2

Initial freeze, 162 ood + 63 in_domain, sha256
`5527caa5e3ce9e0dba76a94b835fb5d7e41726cdde01217a97f1cacb471d0682`.

### 1.0a — 4 `es` numbers_only rows, commit 350a3c4

After the 1.0 baseline was recorded, four Spanish `numbers_only` rows were replaced:

```
12345                -> doce mil trescientos cuarenta y cinco
98765                -> noventa y ocho mil setecientos sesenta y cinco
2024                 -> dos mil veinticuatro
3.14159              -> coma uno cuatro uno cinco nueve
```

The replacements spell the digits out in Spanish. The originals were bare ASCII digits
labelled `language: "es"`, which is not Spanish text — a row tagged Spanish that contains
no Spanish word. This is recorded here because the edit happened *after* a baseline had
been measured, which is exactly the ordering this file exists to make visible.

Consequence: the `eb7e5d2` baseline numbers were measured on different bytes than the file
being judged. They are kept for reference only. Every baseline quoted in the run table is
re-measured on the 1.1 bytes.

The 1.0a edit was *not* result-driven — it fixed a labelling defect, not a score — but the
ordering was still wrong, and `test_the_eval_set_matches_its_recorded_hash` now makes the
hash the authority rather than anyone's memory.

## Baselines on the 1.1 bytes

Both re-measured after the 1.1 corrections, before any 8-class candidate was scored.
Reproduce with:

```
uv run python -m scripts.measure_ood --classifier transformer --threshold 0.55 --sweep \
    --run-id baseline_transformer_v2
uv run python -m scripts.measure_ood --classifier rules --threshold 0.55 \
    --run-id baseline_rules_v2
```

| metric | transformer @0.55 (7 public intents) | rules @0.55 |
| --- | --- | --- |
| OOD rejection overall | 19/162 = 0.1173 | 133/162 = 0.8210 |
| OOD en | 5/49 = 0.1020 | 40/49 = 0.8163 |
| OOD hi | 8/47 = 0.1702 | 35/47 = 0.7447 |
| OOD es | 4/41 = 0.0976 | 39/41 = 0.9512 |
| OOD hinglish | 2/25 = 0.0800 | 19/25 = 0.7600 |
| in-domain false rejection | 1/89 = 0.0112 | 34/89 = 0.3820 |

The two baselines fail in opposite directions, which is what makes the OOD class worth
trying at all. The transformer accepts almost everything off-topic (0.1173) but almost
never rejects a real request (0.0112). The rules reject most off-topic input (0.8210) but
reject 38.20% of legitimate traffic.

Neither baseline clears the 0.85 overall gate on its own, and no threshold in the sweep
reaches it: the best transformer point is 0.5000 at threshold 0.95, where false rejection
has already risen to 6.74%. The threshold alone is not a lever; a model that can actually
emit `out_of_scope` is.
