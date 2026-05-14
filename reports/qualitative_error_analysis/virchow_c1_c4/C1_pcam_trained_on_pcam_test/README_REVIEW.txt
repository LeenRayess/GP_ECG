1) Open gallery.html in a browser (images + captions).
2) Open review_labels_template.csv in Excel/Sheets.
3) For each case_id, mark checklist items Present / Absent / Unclear
   (exact rubric + definitions: docs/qualitative_error_analysis_protocol.md section 6).
4) Rows match gallery.html top-to-bottom (bucket sections alphabetically;
   review_order is 1..N in that sequence). Randomize your review in Excel if you prefer.
5) After finishing, save as review_labels_completed.csv in this folder.

Condition ID (for thesis): C1
Direction label: PCam_test (in-domain)

If raw images are missing, only preprocessed PNGs were written; pass raw_test_x
with the original split test_x.h5. Row alignment uses manifest.json kept_indices
(next to preprocessed test_x.h5) so each preprocessed row maps to the correct raw patch.