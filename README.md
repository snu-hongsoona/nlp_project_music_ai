

## 1. Preparing datasets

### 1.1 Pre-processing datasets

- process `total.csv` file to json file.

    ```bash
    python map_midi_to_label.py
    ```
    - File `midi_label_map_apex_reg_cls.json` is generated.
    - Currently, peak value from kernel density estimation is used as label.

- Generate XAI for music dataset in OctupleMIDI format using the midi to label mapping file with `gen_xai.py`.

    ```bash
    python -u gen_xai.py
    ```
    - Currently, train / test set is splitted by segments

- Binarize the raw text format dataset. (this script will read `xai_data_raw_apex_reg_cls` folder and output `xai_data_bin_apex_reg_cls`)

    ```bash
    bash scripts/binarize_xai.sh xai
    ```


## 2. Training

* **Download our pre-trained checkpoints here: [small](https://msramllasc.blob.core.windows.net/modelrelease/checkpoint_last_musicbert_small.pt) and [base](https://msramllasc.blob.core.windows.net/modelrelease/checkpoint_last_musicbert_base.pt), and save in the ` checkpoints` folder. (a newer version of fairseq is needed for using provided checkpoints: see [issue-37](https://github.com/microsoft/muzic/issues/37) or [issue-45](https://github.com/microsoft/muzic/issues/45))**


### 2.1 Pretrain on XAI music classification task

- you should modify hyperparameters, checkpoint path, etc in sh file.

    ```
    bash scripts/classification/pretrain.sh
    ```

### 2.2 Fine Tuning on XAI music classification task

- you should modify hyperparameters, checkpoint path, etc in sh file.

    ```
    bash scripts/classification/finetuning.sh
    bash scripts/classification/finetuning_only.sh
    ```
finetuning.sh file will finetune every pretrained model generated by step 2.1. finetuning_only.sh will finetune no pretrained model. 