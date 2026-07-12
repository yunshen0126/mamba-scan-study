# Stage1C CIFAR-10 Visual Findings Review

## Scope and Method

This review compares the seed-0 Mamba checkpoints for `row`, `same_row_4`, and
`real_4dir`. Heatmaps are Grad-CAM-style attributions on the final 8x8 `feat2d`
map, all targeting the true-class logit. Each positive attribution map is
normalized to unit spatial mass before computing `real_4dir - same_row_4`.
Therefore, the difference panels describe where spatial attribution moved, not
an independently min-max-normalized visual contrast.

The examples are selected by model behavior over all 10,000 CIFAR-10 test
images, not sampled at random.

## Checkpoint Results

| Variant | Best accuracy | Best epoch | Branches |
|---|---:|---:|---|
| `row` | 75.95% | 30 | row |
| `same_row_4` | 78.11% | 30 | row,row,row,row |
| `real_4dir` | 79.07% | 30 | row,col,diag,anti_diag |

`real_4dir` exceeds `same_row_4` by 0.96 percentage points for this seed, while
both four-branch models remain substantially above the single-row baseline.

## Full-Test Behavior Counts

| Behavior | Test images |
|---|---:|
| row wrong, both four-branch models correct | 578 |
| same_row_4 wrong, real_4dir correct | 769 |
| same_row_4 correct, real_4dir wrong | 673 |
| both correct, real_4dir true-class confidence higher | 3,682 |
| both correct, same_row_4 true-class confidence higher | 3,456 |

The net real-direction advantage over the same-row control is only 96 images.
Across all test images, the mean true-class confidence change is +0.00695 and
the median is +0.00039. The direction-specific advantage is therefore small and
heterogeneous rather than a uniform confidence shift.

## Human-Reviewed Cases

### Both four-branch models fix the row baseline

- Index 5135, automobile: `same_row_4` and `real_4dir` both move substantial
  attribution onto the lower vehicle/body region and both classify correctly.
  Their main object response is similar. The largest positive real-minus-same
  region is in the upper-right background, so this case supports a
  multi-branch/capacity explanation more than a direction-specific one.
- Index 6395, ship: both four-branch models are correct. `real_4dir` allocates
  somewhat more attribution to the right side and horizontal extension of the
  ship/horizon, while `same_row_4` already captures the central ship evidence.
  This is a weak candidate for a directional spatial contribution, but it is
  not clean enough to establish the mechanism by itself.

### same_row_4 wrong, real_4dir correct

- Index 9113, automobile: confidence changes from 0.002 to 0.981, but the
  strongest added attribution is around the lower-left shadow/image boundary,
  not a clearly new vehicle contour or long-range object structure. The
  prediction improvement is real; the heatmap does not provide a clean spatial
  explanation for it.
- Index 4285, automobile: confidence changes from 0.004 to 0.929. The largest
  positive real-minus-same region is above the car roof, partly in background.
  `same_row_4` already emphasizes the roof/body boundary. This case does not
  support a strong claim that real directions recover uniquely useful object
  geometry.

### Both correct, real_4dir more confident

- Index 8957, horse: both maps cover the horse body. `real_4dir` expands
  attribution toward the lower-left body/leg region, but the change is diffuse
  and `same_row_4` already uses the main subject. This is at most weak visual
  support for direction-specific structure.
- Index 180, airplane: `real_4dir` adds attribution near the lower-right
  wing/tail extension while reducing some upper-image response. This is one of
  the clearer examples where the real-direction model appears to cover a
  spatially extended object part missed by the control. It should be reported
  as a representative possibility, not as a general result.

### Counterexamples

- Index 6285, ship: `same_row_4` is correct with 0.956 true-class confidence,
  while `real_4dir` is wrong with 0.018. The real-direction map shifts mass
  toward upper background/sky and unrelated lower regions instead of keeping
  the coherent ship-body response shown by the control.
- Index 2554, deer: `same_row_4` is correct with 0.900 confidence, while
  `real_4dir` predicts horse with only 0.021 true-class confidence. The strongest
  positive difference is in the upper-right background/adjacent structure,
  providing a direct counterexample to a uniformly beneficial direction effect.

## Branch-Level Evidence

For the three branch figures, mean pairwise cosine similarity between the
row/col/diag/anti-diag attribution maps is 0.627, 0.561, and 0.539. The branches
are not identical and do encode visibly different spatial responses. However,
the differences are not consistently complementary object parts: some branch
hotspots fall on background, borders, or shadows. Branch diversity is visible,
but these figures do not show that directional diversity is the primary cause
of the accuracy gain.

## Paper-Safe Conclusion

The visual analysis is consistent with the numeric Stage1C result. Four full
branches clearly improve over one row branch. `real_4dir` sometimes captures an
extended object region that `same_row_4` underweights, but equally strong
counterexamples exist, and most of the four-branch gain is already reproduced
by `same_row_4`. The current evidence supports a multi-branch/capacity effect
with a possible small, sample-dependent directional contribution. It does not
support the strong statement that scan direction is the main source of the
CIFAR-10 improvement.

## Limitations

- The qualitative analysis uses one seed.
- The final feature map is only 8x8, so localization is patch-level rather than
  pixel-level.
- Grad-CAM is correlational and does not establish causal contribution.
- Wrong-prediction panels target the true-class logit for a controlled
  cross-model comparison; they do not visualize the predicted-class rationale.

No Stage2 attribution experiment was performed.
