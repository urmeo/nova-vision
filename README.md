# NovaVision

A Python prototype for turning English text into image prompts with emotion, palette, and lighting cues. Separate experiments test whether an image probe recovers the intended emotion.

<a href="docs/assets/pipeline.svg"><picture><source media="(max-width: 900px)" srcset="docs/assets/pipeline-mobile.svg"><img src="docs/assets/pipeline.svg" alt="NovaVision pipeline: write text, analyze text, build a prompt, and render an image. Separate experiments set known targets, reuse prompt building and rendering for four paired tiers, then compare image-probe estimates with the targets. Probe outputs are model estimates, not human ratings."></picture></a>

[Editable Figma source](https://www.figma.com/design/6lh7NsSYszXlyDrk6I1IaU?node-id=24-2) · [Figure sources and captions](docs/pipeline-figures.md).

[https://urmeo.github.io/nova-vision/](https://urmeo.github.io/nova-vision/) · [Video](#video) · [Run locally](#run) · [Notebook](notebooks/reproduce.ipynb)

Keywords: **Python · PyTorch · Transformers · Diffusers · NumPy · Pillow · Flask**

The [public website](https://urmeo.github.io/nova-vision/) is a static preview. Run the Python server for text analysis and image generation.

## Video

**Interface and pipeline · 0:28**

<a href="https://urmeo.github.io/nova-vision/assets/videos.html#pipeline-tour"><img src="docs/assets/pipeline-interface.png" alt="Watch the interface tour showing the static preview, pipeline, and saved charts" width="960"></a>

[Watch video](https://urmeo.github.io/nova-vision/assets/videos.html#pipeline-tour) · [Download MP4](docs/assets/pipeline-tour.mp4)

## Run

```bash
git clone https://github.com/urmeo/nova-vision.git
cd nova-vision
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[app,ml]"
python -m pip install -e ".[dev]"
python -m pytest
node --test tests/test_web_ui.js
python scripts/visualize.py
NOVA_BACKEND=diffusers python server.py
```

[http://127.0.0.1:8000](http://127.0.0.1:8000).

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell, then set `$env:NOVA_BACKEND = "diffusers"` before running `python server.py`.

## Interface

| Current local interface | Earlier text-analysis example |
|:---:|:---:|
| <a href="docs/assets/interface.png"><img src="docs/assets/interface.png" alt="Current NovaVision interface with text input, style selection, and an empty image panel" width="420"></a> | <a href="outputs/figures/emotion_analysis.png"><img src="outputs/figures/emotion_analysis.png" alt="Earlier interface showing text emotion scores, affect estimates, and a generation prompt" width="420"></a> |

| Earlier prompt details | Earlier displayed image |
|:---:|:---:|
| <a href="docs/assets/prompt-example.png"><img src="docs/assets/prompt-example.png" alt="Earlier interface showing the New adventure input and the longer prompt assembled from it" width="420"></a> | <a href="docs/assets/generation-example.png"><img src="docs/assets/generation-example.png" alt="Earlier interface showing the New adventure input, text emotion scores, and a sunlit meadow image" width="420"></a> |

## Results

| Recovery and 95% confidence intervals | Raw-tier confusion |
|:---:|:---:|
| <a href="docs/assets/results/recovery.png"><img src="docs/assets/results/recovery.png" alt="Saved pilot recovery accuracy with item-bootstrap 95 percent confidence intervals and chance reference" width="420"></a> | <a href="outputs/figures/confusion_raw.png"><img src="outputs/figures/confusion_raw.png" alt="Raw-tier confusion matrix comparing intended and recovered emotion labels" width="420"></a> |

| Emotion-tier confusion | Affect-tier confusion |
|:---:|:---:|
| <a href="outputs/figures/confusion_emotion.png"><img src="outputs/figures/confusion_emotion.png" alt="Emotion-tier confusion matrix from the saved pilot" width="420"></a> | <a href="outputs/figures/confusion_affect.png"><img src="outputs/figures/confusion_affect.png" alt="Affect-tier confusion matrix from the saved pilot" width="420"></a> |

[![Emotion-tier confusion counts](docs/assets/results/confusion.png)](docs/assets/results/confusion.png)

| Separate scene diagnostic | Affect-tier valence and arousal |
|:---:|:---:|
| <a href="outputs/figures/confusion_scene.png"><img src="outputs/figures/confusion_scene.png" alt="Confusion matrix for the seven fixed-scene diagnostic images" width="420"></a> | <a href="outputs/figures/va_affect.png"><img src="outputs/figures/va_affect.png" alt="Intended prior values versus image-probe estimates of valence and arousal in the affect tier" width="420"></a> |

### Probe checks

[![Saved CLIP accuracy on labelled face and scene samples](docs/assets/results/probe-comparison.png)](docs/assets/results/probe-comparison.png)

### Planning curve

[![Illustrative sample-size curve connecting the saved binomial simulation points](docs/assets/results/planning-curve.png)](docs/assets/results/planning-curve.png)

Sole contributor: [urme](https://github.com/urmeo).

## References

Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A dataset of fine-grained emotions. In D. Jurafsky, J. Chai, N. Schluter, & J. Tetreault (Eds.), *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 4040–4054). Association for Computational Linguistics. [https://doi.org/10.18653/v1/2020.acl-main.372](https://doi.org/10.18653/v1/2020.acl-main.372)

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. In M. Meila & T. Zhang (Eds.), *Proceedings of the 38th International Conference on Machine Learning* (Vol. 139, pp. 8748–8763). PMLR. [https://proceedings.mlr.press/v139/radford21a.html](https://proceedings.mlr.press/v139/radford21a.html)

Sauer, A., Lorenz, D., Blattmann, A., & Rombach, R. (2023). *Adversarial diffusion distillation* [Preprint]. arXiv. [https://doi.org/10.48550/arXiv.2311.17042](https://doi.org/10.48550/arXiv.2311.17042)

Yang, J., Huang, Q., Ding, T., Lischinski, D., Cohen-Or, D., & Huang, H. (2023). *EmoSet: A large-scale visual emotion dataset with rich attributes* [Preprint]. arXiv. [https://doi.org/10.48550/arXiv.2307.07961](https://doi.org/10.48550/arXiv.2307.07961)

## License

[MIT](LICENSE)
