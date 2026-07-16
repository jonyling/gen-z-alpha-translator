# Filtered slice of `genz_dataset_augmented_1500.csv`

Auto-filter only — **you still need a manual pass**.

| File | Rows | Purpose |
|---|---:|---|
| `genz_dataset_augmented_1500.csv` | 1665 | Original (untouched) |
| `genz_dataset_augmented_1500.filtered.csv` | 97 | Candidates for manual keep/edit |
| `genz_dataset_augmented_1500.dropped.csv` | 1568 | Removed rows + `drop_reasons` |

## Auto-drop rules
- `source == generated` or `trend_status == generated`
- Gloss / dictionary English (`extremely`, `somewhat secretly`, `X or Y`, `a very specific`, …)
- `af` → `extremely` substitution
- Multi-word meaning pasted verbatim into `normal_sentence`
- Identical slang/normal, normal==term only, obvious Mad Libs frames
- Slang term left untranslated in the English side (with small allowlist)

## Kept breakdown
- by source: {'augmented': 76, 'original': 21}
- by trend_status: {'trending': 46, 'real': 34, 'viral': 17}

## Drop reason counts
- 1150: gloss_in_meaning
- 1032: meaning_pasted_into_normal
- 1028: gloss_in_normal
- 486: af_to_extremely
- 358: bad_slang_frame
- 351: strat_generated
- 188: term_untranslated_in_normal
- 150: identical_slang_normal
- 20: source_generated

## Suggested manual pass
1. Open `genz_dataset_augmented_1500.filtered.csv` in Excel/Sheets.
2. Delete rows that still sound unnatural.
3. Optionally rewrite `normal_sentence` to natural English (keep `slang_sentence` if the slang is real).
4. When happy, either rename to a new raw file and add a `SOURCES` entry in `src/config.py`, or replace usage of the augmented file.
