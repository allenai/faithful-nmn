{
  local lxmert_model_path = std.extVar("LXMERT_model_path"),
  local image_feat_path = std.extVar("path_to_vg_gqa_obj36.tsv"),

  "dataset_reader": {
    "type": "gqa",
    "topk": -1,
    "image_feat_path": image_feat_path,
    "cache_path": std.extVar("cache_path"),
    "reload_tsv": false,
    "positive_threshold": 0.75,
    "negative_threshold": 0.25,
    "object_supervision": true,
    "require_some_positive": false,
    "relations_path": "/net/nfs.corp/allennlp/data/visual_genome_100K/relationships.json",
    "attributes_path": "/net/nfs.corp/allennlp/data/visual_genome_100K/attributes.json",
    "objects_path": "/net/nfs.corp/allennlp/data/visual_genome_100K/objects.json"
  },
  "train_data_path": "dataset/train_balanced_exist_norelation_questions.json",
  "validation_data_path": "dataset/val_balanced_exist_norelation_questions.json",
  // "train_data_path": "dataset/train_balanced_exist_questions.json",
  // "validation_data_path": "dataset/val_balanced_exist_questions.json",
  "model": {
    "type": "gqa_end_to_end_module_network",
    "encoder": {
      "type": "lxmert",
      "max_seq_length": 42,
      "pretrained_model_path": lxmert_model_path,
    },
    "nmn_settings": {
        "mask_non_attention": true,
        "use_sum_counting": true,
        "use_count_module": false,
        "filter_find_same_params": true,
        "use_cross_encoding": false
    },
    "use_gold_program_for_eval": true,
    "denotation_loss_multiplier": 1.0,
    "object_loss_multiplier": 1.0,
  },
  "iterator": {
    "type": "basic",
    "batch_size": 16
  },
  "trainer": {
    "optimizer": {
        "type": "bert_adam",
        "lr": 0.00001,
        "warmup": 0.1,
        "t_total": 109995*4/16,
    },
    "grad_norm": 5.0,
    "validation_metric": "+denotation_acc",
    "num_serialized_models_to_keep": 1,
    "num_epochs": 4,
    "cuda_device": 0
  }
}
