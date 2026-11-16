# Submission handoff — 16 November 2026

## Included in this repository

- Reviewed implementation and 20 unit tests.
- Google Colab small-mode notebook at `notebooks/colab_pipeline.ipynb`.
- Real Colab-generated `reports/` JSON, NDCG plot and `docs/RESULTS.md` (source: uploaded `results_bundle.zip`).
- API documentation, architecture diagram, measured-run verification notes and presentation deck.
- Triton configuration/metadata **without model weights**; Airflow DAG source.

## Important distinctions

- **Small run completed.** `MODE=full` (MovieLens 25M) is still an optional, larger run; do not claim it was done.
- **Working in-process API demo**, not a publicly hosted service. The output in Colab showed Redis caching and a cold-start response.
- **Triton exported, not serving-tested.** Model binaries are not in the ZIP or GitHub repository, and cannot be regenerated without retraining or retaining the Colab model files.
- **A/B replay is simulated**, not an online trial. No real-user CTR was measured.
- A screenshot or screen recording of the successful Colab `/recommend` cell can document the demonstration.
- `docs/PRESENTATION_OUTLINE.md` and `docs/Recommendation_Engine_10min.pptx` use only the completed small-run findings.

## GitHub submission

Publish the **contents of the `recsys-engine` directory**, not the outer ZIP. `.gitignore` excludes raw data, model weights, temporary files and archives, while allowing reports and the NDCG image. Suggested terminal steps, from **inside the extracted `recsys-engine` folder** after creating an empty GitHub repository:

```bash
git init
git add .
git commit -m "Add verified small-mode recommender and results"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL; do not literally paste that placeholder. Authentication may require your GitHub login or token. Verify the repo online before submitting its URL.

## Extra evidence / remaining tasks

- Capture a short screen recording of the successful in-process Colab API cell (personalised recommendation, cached second request, cold start).
- If your grading rubric requires it and runtime/resources permit, perform `MODE=full` on Colab and regenerate the result files; **replace** small-mode numbers in the presentation if you do that.
- Do not claim live Triton, Airflow scheduling or real A/B traffic unless separately executed and logged.
- Write the personal-reflection section yourself; prompts are provided, but no experiences were invented.
