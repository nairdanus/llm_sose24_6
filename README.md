# Reinforcement Learning from Human Feedback for Context-Aware Hate-Speech Discrimination

This repostitory provides a method to fine-tune GPT-2 to be more context-sensitive when detecting hate-speech by Reinforcement Learning Human Feedback

Reward model can be trained or accessed directly from https://huggingface.co/nairdanus/appraising_hate_speech

Policy model can be found at https://huggingface.co/nairdanus/gpt2-rlhf-finetuned-hate

## Dataset

The HateWic Dataset containing contextually sensitive senses was adapted to be used for RLHF

## Run Code

### Training the Model
Please refer to RLHF_with_Custom_Datasets.ipynb. After installing the necessary trlx repository, replace the following files with the respective files in src/trlx_adjustments/:
- trlx/trlx/models/modeling_base.py
- trlx/trlx/trainer/accelerate_base_trainer.py
- trlx/trlx/trainer/accelerate_ppo_trainer.py

### Evaluating the model
Please refer to lime.ipynb.
