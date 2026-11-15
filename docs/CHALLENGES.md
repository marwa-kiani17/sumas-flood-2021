# Problems we hit while building this project (and how each was solved)

This is an honest log of what went wrong during development. Good to tell the professor: it shows that results
were checked, not just accepted. Each row follows the same pattern: **what happened, why, what we did, what we
learned.**

## A. Setup and tooling

| # | What happened | Why | Fix | Lesson |
|---|---|---|---|---|
| 1 | The assistant could not find `conda` after you installed it. | The coding session was started before the install, so its PATH was out of date. | You ran the commands yourself in a fresh terminal; later the environment's own Python was called with its folders added to PATH. | A new install is invisible to programs that were already running. |
| 2 | PyTorch printed `cuda False`. | `segmentation-models-pytorch` installed a CPU-only PyTorch first, so the CUDA build was skipped. | Force-reinstalled the CUDA build (`cu128`); now the RTX 3060 is used. | Check `torch.cuda.is_available()` right after installing. |
| 3 | A crash `0xc06d007f` when running tests with the env's Python directly. | Running `python.exe` without activating the env, so Windows could not find the math libraries (DLLs). | Added the env folders to PATH for that command. | `conda activate` does more than it seems: it sets library paths. |
| 4 | OpenMP error "libomp.dll ... libiomp5md.dll already initialized". | PyTorch and NumPy each bundle their own OpenMP library on Windows. | Set `KMP_DUPLICATE_LIB_OK=TRUE` in the CV scripts (a standard workaround). | Known Windows quirk, not a bug in our code. |
| 5 | Reading the PDF printed a character-encoding error. | The Windows console used an old encoding that cannot print the arrow symbol. | Ran Python in UTF-8 mode. | Always use UTF-8 for text output. |
| 6 | Pretrained model weights would have been saved in the user's home folder. | The libraries cache downloads in a default location outside the project. | Redirected the caches into `data/cache/` inside the project (your rule: nothing outside the folder). | Check where libraries write files. |

## B. Data

| # | What happened | Why | Fix | Lesson |
|---|---|---|---|---|
| 7 | Several expected gauges had no 2021 data. | Coquihalla near Hope ended in 1983; Tulameen near Princeton had only about 1 year of record. | Searched by name, checked each candidate's years and Nov 2021 coverage, and used the active neighbouring stations. | Never trust a station name: check the dates. |
| 8 | The weather service answered `429 Too Many Requests` again and again. | Open-Meteo's free tier limits the amount of data per minute, per hour and per day, and long hourly requests cost a lot. | Cached every response, made the script resumable, waited out the per-minute limit automatically, trimmed unused variables, downloaded rivers in order of importance, and finished the last river on the next day. | Plan for rate limits: cache everything and make downloads restartable. |
| 9 | The Kettle River could never get flood thresholds. | It is a seasonal gauge (no winter data), so no year had enough complete days. | Dropped it (it also had no flood in Nov 2021). | A fix that needs a lot of special handling for one river is not worth it. |
| 10 | The Nooksack gauge that flooded Sumas Prairie had no long record. | The gauges closest to the overflow have short records (7-19 years). | Used the downstream Ferndale gauge (1980-2023). It is below the overflow point, which is noted as a limit. | Long records beat convenient locations for training. |
| 11 | Two weather columns are mostly empty. | The first 4 rivers were downloaded before we trimmed the variable list. | The model does not use those columns. | Changing a download halfway leaves mixed data: document it. |

## C. Statistics and models

| # | What happened | Why | Fix | Lesson |
|---|---|---|---|---|
| 12 | Flood levels were absurd (a 25-year flood of millions of m3/s) or infinite. | With only 20-30 yearly peaks, an unconstrained GEV fit picks extreme tail shapes. | Bounded the shape parameter to a realistic range and capped return periods at 10,000 years. | Small samples need constrained fits. |
| 13 | Models stopped exactly at the maximum number of trees. | The learning rate was too low for the tree limit. | Raised the learning rate and the limit. | If early stopping never triggers, the model is under-trained. |
| 14 | **The model under-predicts the biggest Nov 2021 peaks** (for example, 360 forecast vs 932 m3/s observed on the Similkameen). | Tree models cannot predict values beyond what they saw in training, and Nov 2021 broke records. | Tried linear-leaf trees (modest gain in warnings, no gain in peak size) and documented the limit. | Report a negative result honestly; it is a real finding. |
| 15 | The rain-on-snow and "AR-like storm" features did not improve scores. | The basic rain, temperature and snow-depth features already contain the same information. | Reported it as a negative result in the ablation. | Not every idea works. |
| 16 | Warning ranges (10-90%) covered only about 72-75% of observations, not 80%. | Model uncertainty is under-estimated, especially for extremes. | Documented it; a calibration step is future work. | Always check coverage, not just accuracy. |
| 24 | After adding the 11th river (Salmon), the pooled warning scores showed almost no gain over persistence at 3 days, although the earlier 10-river result looked strong. | 142 of the 162 test flood days are slow snowmelt floods (72 from the Salmon alone), where "tomorrow = today" already works. | Split the scores by river type: on rain-driven and mixed rivers the model wins clearly (3 days: POD 0.73 / 0.56 versus 0.00), on snowmelt rivers persistence is as good or better. Corrected the README. | A pooled score can hide where a model helps. Always break results down, and re-check claims when data is added. |

## D. Satellite mapping

| # | What happened | Why | Fix | Lesson |
|---|---|---|---|---|
| 17 | The first flood map showed the Fraser River as "new flood". | The "before" image missed parts of the river (wind-roughened water looks bright), so the river appeared when it calmed. | A first fix (water seen in half the scenes) did not work and removed real flood patches. The final fix marks pixels that are dark in at least 80% of the full-coverage scenes as permanent water. | Look at the maps, not just the numbers. |
| 18 | Four satellite scenes covered only 61% of the area. | They come from a different satellite (Sentinel-1A) with a different footprint. | Reported areas on the region seen by every scene, so the series is comparable. | Compare like with like. |
| 19 | The big flooded patch is only partly mapped. | The U-Net was trained on floods worldwide, not on BC farmland. | Documented as a limit; the threshold method (Otsu) is shown for comparison. | Models trained elsewhere may not transfer. |
| 20 | Validation score jumped around (0.35 to 0.64) during training. | Small validation set and a high learning rate early on. | Kept the best epoch and ran extra seeds to show the spread. | One run is not enough; show variation. |

## E. Process mistakes (ours, and fixed)

| # | What happened | Fix |
|---|---|---|
| 21 | The assistant tried to write a temporary file outside the project folder and use the system Python, against your rules. The tool blocked it. | Rewrote the step so it runs only inside the project and the conda env. |
| 22 | A plotting formula was written wrongly (an `and` between two arrays) before it ever ran. | Caught on review and fixed before running. |
| 23 | The first `git push` failed with "repository not found". | The URL still contained the placeholder `YOUR-USERNAME`; replaced with the real one. |
| 25 | A command accidentally started the system Python (not the project environment) with an empty script and hung. | A mistake in a shell command. | Stopped it at once; from then on all runs used only the `floodml` environment and wrote logs inside the project (`logs/`). |

## What to say to the professor

"I built it in steps and checked each step against the real data. Several first attempts were wrong or misleading
(absurd flood levels, the Fraser River counted as flood, peaks under-predicted). I fixed what could be fixed and
reported what could not. The main open problem is predicting record-breaking floods with a tree model; the next step
would be a model that extrapolates, such as an LSTM, and real atmospheric river inputs."
