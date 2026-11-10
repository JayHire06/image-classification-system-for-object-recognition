# ES 335 individual defense prep

Use these prompts to practice explaining the study in your own words. Be
specific about the decisions, checks, or implementation you personally
performed. The project website discloses where Codex assisted.

1. **What question did the experiment answer?** Compare the full HOG + SVM
   and MobileNetV2 pipelines on ten-class CIFAR-10 recognition. The input is
   one image and the target is one dataset class. Explain why the comparison
   does not isolate the contribution of ImageNet pretraining.
2. **How did you keep model selection separate from final scoring?** Draw
   the 40,000/10,000/10,000 split. Explain the SVM's internal 32,000/8,000
   selection split, the separate calibration data, and CNN checkpoint
   selection on validation. Disclose that the initial baseline test score
   had been viewed before the training-only tuning revision.
3. **Why these transformations and metrics?** Explain local edge
   histograms, block normalization, the pretrained CNN's input
   normalization, and why random transforms belong in training. Explain
   how macro-F1, accuracy, and the confusion matrix answer different
   questions.
4. **What did the results reveal?** Use the site's model and class controls
   to compare the cat row. Explain why 83 cat-to-dog errors for MobileNetV2
   and 170 for HOG + SVM are more specific than an overall score. Explain
   why a high-confidence mistake can still occur.
5. **What does the uncertainty estimate cover?** The paired bootstrap
   resamples test images for both models together. Its interval describes
   variation from resampling those images with fixed trained models.
   Retraining and shifts in image source need separate experiments.
6. **What if one choice changed?** Pick one change, such as training
   MobileNetV2 without ImageNet weights. Explain which settings and split
   you would hold fixed, what result you expect, and what evidence would
   change your mind. Describe the expectation as a hypothesis.

Practice rerunning a command from the README and explaining which
artifacts it produces.
