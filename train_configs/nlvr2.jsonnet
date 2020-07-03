local lxmert_model_path = "lxmert_pretrained/model";
local image_feat_path = "dataset/nlvr2";
local cache_path = std.extVar("cache_path");
local annotations_path = "dataset/nlvr2/all_annotations_round4.tsv";
local tiny = false;
local num_epochs = 50;
local batch_size = 16;
local num_training_samples = 30000;
local num_boxes = 36;

{
  "dataset_reader": {
    "type": "nlvr2",
    "image_feat_path": image_feat_path + '/train_obj36.tsv',
    "annotations_path": annotations_path,
    "tiny": tiny,
    "cache_path": cache_path,
    "only_with_annotation": true,
    "reload_tsv": false,
    "max_boxes": num_boxes,
    "ignore_modules": [],
    "max_seq_length": 13
  },
  "validation_dataset_reader": {
    "type": "nlvr2",
    "image_feat_path": image_feat_path + '/valid_obj36.tsv',
    "annotations_path": annotations_path,
    // "box_annotations_path": "dataset/nlvr2/nlvr2_annotated_boxes.csv",
    "tiny": tiny,
    "cache_path": cache_path,
    "only_with_annotation": true,
    "reload_tsv": false,
    "max_boxes": num_boxes,
    "ignore_modules": []
  },
  "train_data_path": "dataset/nlvr2/train.json",
  "validation_data_path": "dataset/nlvr2/valid.json",
  "model": {
    "type": "nlvr2_end_to_end_module_network",
    "freeze_encoder": false,
    "encoder": {
      "type": "lxmert",
      "max_seq_length": 60,
      "pretrained_model_path": lxmert_model_path,
    },
    "dropout": 0.1,
    "use_gold_program_for_eval": true,
    "use_modules": true,
    "nmn_settings": {
        "use_sum_counting": true,
        "use_count_module": true,
        "mult_attended_objects": false,
        "find_dot_product": false,
        "find_dot_product_vis_query": false,
        "find_qst_query_num_layers": 1,
        "find_dot_activation": "relu",
        "find_dot_dropout": false,
        "filter_find_same_params": true,
        "max_boxes": num_boxes,
        "mask_non_attention": true,
        "simple_with_relation": false,
        "use_cross_encoding": false
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": batch_size
  },
  "trainer": {
    "type": "callback_save_predictions",
    "num_epochs": num_epochs,
    "cuda_device": 0,
    "optimizer": {
      "type": "bert_adam",
      "lr": 0.00001,
      "warmup": 0.2,
      "t_total": num_training_samples*num_epochs/batch_size,
    },
    "callbacks": [
        "log_to_tensorboard",
        "validate_save_predictions",
        {
            "type": "checkpoint",
            "checkpointer": {
                "num_serialized_models_to_keep": 0
            },
        },
        {
            "type": "track_metrics",
            "validation_metric": "+denotation_acc"
        }
    ]
  }
}
