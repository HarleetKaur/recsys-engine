# Submission handoff — updated 20 November 2026

## Final local verification — 20 November 2026

- GitHub contains the project source, notebook, documentation, and committed results.
- The working Docker API uses Keras ranking with FAISS and Redis; personalized and cold-start requests were tested.
- Both Triton models passed direct inference tests, but API-to-Triton integration remains incomplete.
- The separate trained-model backup ZIP passed its archive integrity test. Trained weights are not included in the GitHub repository.
- Presentation and demo recording should be supplied separately if required by the submission instructions.


**GitHub:** https://github.com/HarleetKaur/recsys-engine

## Files to submit or show

- Source, unit tests and reviewed Colab notebook in GitHub; local 21-test suite includes the cold-start cache-isolation regression test.
- Preserved **small** verified run: `docs/RESULTS.md`, root `reports/`, `reports/ndcg.png`.
- **MovieLens 25M-source, 2,000-user subset**: `docs/RESULTS_ML25M_SUBSET.md`, `reports/ml25m_subset_2000/`, `docs/RUN_COMPARISON.md`, `docs/RUN_PROVENANCE_ML25M_SUBSET.json`.
- Updated presentation: `docs/Recommendation_Engine_10min.pptx` and `.pdf` plus `docs/PRESENTATION_OUTLINE.md`.
- Architecture diagram (`docs/architecture.png`), API documentation (`docs/API.md`), and technical verification notes.
- A separate 6-second screen recording of the in-process Colab `/recommend` demo was shown in QuickTime. **The recording is on the user's Mac, not in this archive.** If needed by your reviewer, share or add it separately.

## The exact evidence

- The 25M-source run used `data.max_users=2000`, **297,358** ratings across all splits, and 35 evaluated users. It **did not** train on all 25 million ratings.
- Retrieval-only NDCG@20 **0.0959**, DCN **0.0891**, Wide & Deep **0.0878** on this subset. The deep rankers were **not** better on this metric.
- A/B replay simulated clicks; it was **not** an online A/B test. Triton export metadata does **not** prove live Triton serving or contain model weights. Airflow source does **not** prove scheduling ran.

## Remaining actions for the user

1. Merge the update ZIP from this chat into the existing local Git repository **without removing** the small-run report files.
2. Review the docs and presentation, then commit/push the new report, archive and slide files.
3. Provide the recording separately if required and write a short, truthful personal reflection if the submission requests it.
4. Submit the GitHub URL (and any separately requested presentation/video) by the assigned deadline.

**Optional extensions, not required to describe the executed subset run accurately:** train on all of MovieLens 25M if time/resources allow; validate Triton on compatible x86 Linux; run an Airflow scheduler; conduct a real-user experiment. Do not label any of these completed without direct evidence.

## Latest runtime results — 20 November 2026

FastAPI-to-Triton inference was successfully verified for both
ranking models. The API returned personalized recommendations,
and each model's inference count increased by 200.
A/B reporting works, but real-user A/B testing is not established.
Airflow scheduler execution remains unverified.
