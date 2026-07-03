
conda activate gesturelsm
with-proxy python demo.py -c configs/shortcut_rvqvae_128_hf.yaml



Things to save for restart OD:
# 1. Conda env
# 2. see .gitignore
# 3. Download the model for Demo
```~/.cache/huggingface/hub/models--openai--whisper-tiny.en/```
# 4. Download the model for inference
english_us_arpa
/home/ekkasit/Documents/MFA/pretrained_models/acoustic/english_us_arpa.zip
/home/ekkasit/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict
# 5. VSCode Settings
/home/ekkasit/.vscode-remote/data/Machine/settings.json



## run Demo
```
with-proxy python demo.py -c configs/shortcut_rvqvae_128_hf.yaml
```
### in local machine
```ssh -L 7860:localhost:7860 ekkasit@devvm24589.ldc0.facebook.com```

### Run generation with HTML visualization
with-proxy python demo_html.py \
    --audio demo/examples/2_scott_0_5_5.wav \
    --output generation.html \
    --config configs/shortcut_rvqvae_128_hf.yaml



# train rvq
bash train_rvq.sh


############# Generate ##########################
## first time
```
# $unset HF_HUB_OFFLINE
# $with-proxy python gen.py
```
## After that
```
export HF_HUB_OFFLINE=1
python gen.py
```

## Gen from Trimmed
python gen.py --audio visualize/input/2_scott_0_1_1_128f.wav --out visualize/output_128f
#### with control
python gen.py --mode diffusion --audio visualize/input/2_scott_0_1_1_128f.wav --out visualize/output_128f --guidance_freeze_root false
## trim sample
python trim.py


#### Data Representation ####
./data_representation.html




##### Tasks #####
1. Control for long sequence concatenate chunks, the control position is not aligned.
2. guidance_freeze_root ???
3. SMPL model condition for each person -- foot skating better?
4. Eval with GT joint control
