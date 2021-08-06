## Download Pre-trained Weights

### CIRPLANT

[checkpoint download](https://drive.google.com/file/d/1qku0wVZvY-y5Kc5ANFF2HsKDpl56E0PM/view?usp=sharing)
#### Performance
```bash
('recall_top1_correct_composition', 0.20174599378139202),
('recall_top2_correct_composition', 0.3271944510882564),
('recall_top5_correct_composition', 0.5320975843099737),
('recall_top10_correct_composition', 0.691580961492466),
('recall_top50_correct_composition', 0.9306386032049749),
('recall_top100_correct_composition', 0.9681894283664195),
('recall_subset_top1_correct_composition', 0.4038985888543411),
('recall_subset_top2_correct_composition', 0.6477397751734034),
('recall_subset_top3_correct_composition', 0.8100932791198278)
```

>Note: This version is re-trained. Values differ slightly compared to the paper version due to randomness.

>We limit the number of training epochs to 300. However, continuing training will slightly increase the performance.

#### How-To

Run with the same arguments as in training, but with the additional `--validateonly` and `--load_from_checkpoint` appended at the end.

```bash
python trainval_oscar.py --dataset cirr --usefeat nlvr-resnet152_w_empty --max_epochs 300 --model CIRPLANT-img --model_type 'bert' --model_name_or_path data/Oscar_pretrained_models/base-vg-labels/ep_107_1192087 --task_name cirr --gpus 1 --img_feature_dim 2054 --max_img_seq_length 1 --model_type bert --do_lower_case --max_seq_length 40 --learning_rate 1e-05 --loss_type xe --seed 88 --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss st --batch_size 32 --num_batches 529 --pin_memory --num_workers_per_gpu 0 --comment input_your_comments --output saved_models/cirr_rc2_iccv_release_test --log_by recall_inset_top1_correct_composition --validateonly  --load_from_checkpoint $PATH_TO_CKPT
```

Results will be displayed and saved to the output directory.