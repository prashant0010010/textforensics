# Example inputs (all synthetic)

These files are hand-written or seeded-random **synthetic** texts for trying the interface and for testing the mathematics.
They are not samples of "human" or "AI" writing and say nothing about detector accuracy.

| File | Purpose |
|---|---|
| `academic_with_references.txt` | headings, citations, reference list |
| `technical_with_code_and_table.txt` | code block + table (excluded from analysis by default) |
| `quotations.txt` | block quotations (excluded by default) |
| `punctuation_heavy.txt`, `unicode_mixed.txt` | edge cases |
| `repetitive_synthetic.txt` | heavy phrase/opening repetition; triggers repetition findings |
| `varied_synthetic_2000_words.txt`, `uniform_synthetic_1500_words.txt` | seeded generator output (limited vocabulary) |
| `short.txt` | too short: expect "Insufficient data to assess" |
| `single_paragraph.txt` | one paragraph |
| `generate_long_example.py` | writes a ~30,000-word synthetic file for performance testing |
| `watermark_config.example.json`, `example_scheme.py` | template for the green-list detector (toy rule, not a real scheme) |
