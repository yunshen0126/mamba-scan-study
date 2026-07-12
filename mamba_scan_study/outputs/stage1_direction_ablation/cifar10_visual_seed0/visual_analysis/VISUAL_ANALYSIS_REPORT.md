# Stage1C CIFAR-10 Visual Analysis

This analysis uses Grad-CAM style heatmaps on the final `feat2d` map. All heatmaps target the true CIFAR-10 class logit. Positive attribution is normalized to unit spatial mass per model before computing difference maps. The key comparison is `real_4dir - same_row_4`, because `same_row_4` controls for four full branches without adding new scan directions.

## Checkpoints

- `row`: best_acc=0.7595, best_epoch=29, branch_dirs=`row`
- `same_row_4`: best_acc=0.7811, best_epoch=29, branch_dirs=`row,row,row,row`
- `real_4dir`: best_acc=0.7907, best_epoch=29, branch_dirs=`row,col,diag,anti_diag`

## Selected Cases

### case_row_wrong_four_branch_correct_idx5135

Index 5135 (automobile), category row_wrong_four_branch_correct. The marked positive real_4dir-minus-same_row_4 region is in the top-right of the image. True-class confidence changes from same_row_4=0.988 to real_4dir=0.999 (delta +0.011). Both four-branch models fix the row baseline here, so the figure mainly tests whether the improvement is already explained by multi-branch capacity.

### case_row_wrong_four_branch_correct_idx6395

Index 6395 (ship), category row_wrong_four_branch_correct. The marked positive real_4dir-minus-same_row_4 region is in the middle-right of the image. True-class confidence changes from same_row_4=0.991 to real_4dir=0.995 (delta +0.004). Both four-branch models fix the row baseline here, so the figure mainly tests whether the improvement is already explained by multi-branch capacity.

### case_same_wrong_real_correct_idx9113

Index 9113 (automobile), category same_wrong_real_correct. The marked positive real_4dir-minus-same_row_4 region is in the bottom-left of the image. True-class confidence changes from same_row_4=0.002 to real_4dir=0.981 (delta +0.979). This is the strongest visual test for a possible direction contribution; check whether the positive difference region marks object structure that same_row_4 misses.

### case_same_wrong_real_correct_idx4285

Index 4285 (automobile), category same_wrong_real_correct. The marked positive real_4dir-minus-same_row_4 region is in the top-center of the image. True-class confidence changes from same_row_4=0.004 to real_4dir=0.929 (delta +0.925). This is the strongest visual test for a possible direction contribution; check whether the positive difference region marks object structure that same_row_4 misses.

### case_both_correct_real_higher_confidence_idx8957

Index 8957 (horse), category both_correct_real_higher_confidence. The marked positive real_4dir-minus-same_row_4 region is in the bottom-left of the image. True-class confidence changes from same_row_4=0.325 to real_4dir=0.959 (delta +0.634). Both controls are correct, so only a spatial pattern unique to real_4dir would support a cautious direction-specific interpretation.

### case_both_correct_real_higher_confidence_idx180

Index 180 (airplane), category both_correct_real_higher_confidence. The marked positive real_4dir-minus-same_row_4 region is in the bottom-right of the image. True-class confidence changes from same_row_4=0.297 to real_4dir=0.901 (delta +0.605). Both controls are correct, so only a spatial pattern unique to real_4dir would support a cautious direction-specific interpretation.

### case_real_4dir_negative_same_correct_idx6285

Index 6285 (ship), category real_4dir_negative_same_correct. The marked positive real_4dir-minus-same_row_4 region is in the top-center of the image. True-class confidence changes from same_row_4=0.956 to real_4dir=0.018 (delta -0.938). This is a negative or ambiguous case; it should be used to limit any direction-specific claim.

### case_real_4dir_negative_same_correct_idx2554

Index 2554 (deer), category real_4dir_negative_same_correct. The marked positive real_4dir-minus-same_row_4 region is in the top-right of the image. True-class confidence changes from same_row_4=0.900 to real_4dir=0.021 (delta -0.879). This is a negative or ambiguous case; it should be used to limit any direction-specific claim.

### branch_real_4dir_idx5135

Branch-level Grad-CAM for real_4dir on index 5135. The marked fused high-response region is in the bottom-right. Mean pairwise cosine similarity between branch attribution maps is 0.627. Use this panel to judge whether row/col/diag/anti_diag branches are visually complementary or mostly redundant.

### branch_real_4dir_idx6395

Branch-level Grad-CAM for real_4dir on index 6395. The marked fused high-response region is in the top-center. Mean pairwise cosine similarity between branch attribution maps is 0.561. Use this panel to judge whether row/col/diag/anti_diag branches are visually complementary or mostly redundant.

### branch_real_4dir_idx9113

Branch-level Grad-CAM for real_4dir on index 9113. The marked fused high-response region is in the bottom-left. Mean pairwise cosine similarity between branch attribution maps is 0.539. Use this panel to judge whether row/col/diag/anti_diag branches are visually complementary or mostly redundant.

## Interpretation Rule

If `same_row_4` already shifts attention from background to object regions, the gain should be attributed cautiously to multi-branch capacity or ensemble-like effects. Only cases where `real_4dir` repeatedly highlights spatial structures missed by `same_row_4` support a cautious direction-specific contribution.
