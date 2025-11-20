## CMU 18734/17731 Project Phase 2 and 3 Repo

## Overview
We provide three model sets — each with a target model and its own test set. The MIA (membership inference attack) difficulty levels of the three sets are similar, but you may see small differences in MIA performance across them even when using the same attack code.

We release full details of the train set, including the training data and test labels. This lets you evaluate your method locally and get an estimate of how it will perform when you submit scores for the val and final sets. You can view real-time scores for the validation set, though submissions are time-limited. Final-set scores are only revealed when the final set is opened at the end of the semester and will determine the final rankings. Please develop a general attack method rather than overfitting to the validation set to chase leaderboard performance — the final set decides ranking, and the validation set serves as an additional verification.

## Evaluation
We report **TPR @ 0.01 FPR**.  
- **Phase 2** has a minimum requirement of **0.15** — you must exceed 0.15 to receive that portion of the score.  
- **Phase 3** raises the requirement (not decided yet); we recommend aiming as high as possible (e.g., TPR > 0.35).
- If your MIA techniques for phase 2 already outperform the TPR requirement for phase 3, you may reuse your code for phase 3 submission. 

**Hint:** You have access to the in-distribution data and may use it to train shadow models. Proper use of these data for shadow modeling can greatly improve your MIA score!!!

## Folder structure
- `data/` — contains three subfolders: `train`, `val`, `final`.
  - `train/` includes training data, test data, and test labels.
  - `val/` and `final/` include only test data; your task is to predict membership for these sets.
  - `prepare_data.py` — script used to collect and inspect the training data. Use it to understand the data and to prepare data for shadow models.
- `ft_llm/`
  - `ft_llm.py` — script for fine-tuning a shadow model.
  - `ft_llm_colab.py` — script for fine-tuning a shadow model on colab T4 GPU (T4 does not natively support bf16).
  - `ft_llm.sh` — bash script with our finetuning configuration.
- `models/` — contains three model directories: `train`, `val`, `final`. Use the `train` model for experiments, then predict membership for `val` and `final`.
- `MIA_phase2_3.ipynb` — starter kit for phase 2 and phase 3. Both phases use the same model but have different performance requirements.

## Result submission
participate by your andrew email and submit the zip file (as specificied by notebook) to val phase. The leaderboard is based on codabench, link: https://www.codabench.org/competitions/11238/

FORMAT: after decompressing your zip file, the file structure should look like:
  - val:
    - prediction.csv
  - final:
    - prediction.csv

Please submit by groups after forming an organization (click your name on the top right and create an organization).

Please use andrew email to register only. If you have any questions, email me or post a discussion on canvas.

## Clarification (Newly added 11.3)

As some of you have asked about the details of the project, we confirm some settings here:
 - All three models are trained with LORA, the same setting `-m gpt2 --block_size 512 --epochs 3 --batch_size 8 --gradient_accumulation_steps 1 --lr 2e-4 --lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.05`. The only differences are the training data. (10k training data for each but 3 different sets).
 - You may find out that train phase training data is actually achieved by setting `seed=42` in "data/prepare_data.py". To train other shadow models, remember to change the seed number.
 - I suggest you train your shadow models with the same config to start, once you achieved like 10% TPR at 1% FPR, you may do the hyperparameter tuning and other training configs that you feel reasonable.


## Report 
edit on Nov 12 7pm about the explanation. Reports submitted before do not need changes.

 - About methodology (12 points), we require:
    - a clear diagram of the whole pipeline;
    - the explanation of each component in your diagram (may include how you design shadow models, score calibrations, verification); IMPORTANT: We focus more on the core steps, e.g. how do you design your data splits and train corresponding shadow models, and how do you assign final predictions based on your score functions. Trivial details can be simplified.
    - be concise and structured, attaching necessary plots and be within 3 pages.
 - Results (6 points): show the screenshot of your score of val phase on the leaderboard.
 - Code (2 points): attach the **core** MIA codes.
 - Appendix: you may add more plots here if you want, beyond the 3-page methodology limit

The scores of Results are given by (if your TPR is x):
 - `[0.15,1]` 6
 - `[0.13,0.15)` 6 - 25(0.15-x)
 - `[0.1,0.13)` 5.5 - 50(0.13-x)
 - `[0.05,0.1)` 4-60(0.1-x)
 - `[0.02,0.05)` 1
 - `[0.01,0.02)` 0.5
 - `[0,0.01)` 0


## Phase 3 requirement

we set **the final TPR requirement 0.4 for both val and final** (grades for results will be decided on the lower one). Those who have fulfilled the requirement may only prepare the slides and write up. Others could try with the hints above. and as we announced before, top 3 groups will have bonus points.

### deliverables:
1. **slides for presentation**. We will randomly split you into 7 groups on Monday Dec 1, and 6 groups on Wednesday Dec 3. The detailed ordering will be announced later. you will have 8 min to present and 3 min for QA, overtime will get one point deducted.

2. **final report**. For those who does not change the method and already get full points in phase 2 report, there is no need to submit another report.

**Note**: The report will account for 1/3 and the presentation for 2/3 of the total scores in Phase 3. Please submit a new report only if it includes updated methodology, significant new content, or otherwise meaningful revisions since phase2 submission. If the report remains unchanged, then submission is not required, and the score for your report will constitute the same percentage of your final grade as your presentation. (So please don't submit the same or similar one.)

The content within will be announced later.

The exact deadline is listed on https://www.codabench.org/competitions/11238/#/phases-tabLinks. It is a more precise deadline than which was announced today in class. We will end val phase on Nov 29 11:59pm, and final will last from Nov 30 0:05am to Dec 1 10:30AM. You can only submit one final submission so make sure it's correct. You could download the one with the best val score and submit to final, but it's up to you. We will release the final scores and ranks after Dec 1 10:30AM. If you have made a mistake, email me in time and we won't allow changes after the deadline.

and if you have questions, asks me ASAP to schedule the meeting with me, please don't start at the last minute as I may not be able to help then.

**IMPORTANT**: Please make sure each group only have one submission, remove your submission from the leaderboard if your group already has one! It will mislead the others, and for final ranks, we will only choose the lowest submission if your group submit more than one.

### tips to further improve the performances on Phase3

1. increase the number of shadow models. You could try with different numbers and see the relationship between the number of the performances. Show that in the final presentation if you have. To pass phase 2, only 2 shadow models are sufficient. If you cannot pass with 5 or more shadow models, think about how to do effective data processing and curation.
2. collecting multiple features which all help to further improve, I personally think only very a few features would help, but feel free to try.
3. use quantile regression. Some of you may not fully understand this, here is an implementation that you could refer to. I have verified its effectiveness. Important: quantile regression only works when you have some distinguish scores, then it may improve further. it mainly serves as an efficient method to get rid of shadow models, but combining both may help. Link: https://github.com/iamgroot42/mimir/blob/main/mimir/attacks/quantile.pyLinks. *The input of these models is text, and output is a quantile threshold.* 


