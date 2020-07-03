# pylint: disable=arguments-differ,invalid-name,no-self-use
from typing import List
from types import SimpleNamespace

import torch
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from lib.modules.seq2seq_encoders.lxmert_src.lxrt.entry import (
    LXRTEncoder,
    convert_sents_to_features,
)
from lib.modules.seq2seq_encoders.lxmert_src.lxrt.tokenization import BertTokenizer

args = SimpleNamespace()
setattr(args, "llayers", 9)
setattr(args, "rlayers", 5)
setattr(args, "xlayers", 5)
setattr(args, "from_scratch", False)


@Seq2SeqEncoder.register("lxmert")
class LXMERTEncoder(Seq2SeqEncoder):
    def __init__(
        self,
        max_seq_length: int,
        pretrained_model_path: str,
        num_language_layers: int = args.llayers,
        num_visual_layers: int = args.rlayers,
        num_cross_layers: int = args.xlayers,
    ) -> None:

        super().__init__()
        args.llayers = num_language_layers
        args.rlayers = num_visual_layers
        args.xlayers = num_cross_layers

        self.max_seq_length = max_seq_length
        self.encoder = LXRTEncoder(args, max_seq_length=max_seq_length, mode="lxr")
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

        self.encoder.load(pretrained_model_path)

    def get_embeddings_and_mask(self, sentences: List[List[str]]) -> torch.Tensor:
        sent_features = convert_sents_to_features(
            sentences, self.max_seq_length, self.tokenizer
        )
        input_ids = torch.tensor(
            [f.input_ids for f in sent_features], dtype=torch.long
        ).cuda()
        input_mask = torch.tensor(
            [f.input_mask for f in sent_features], dtype=torch.long
        ).cuda()
        segment_ids = torch.tensor(
            [f.segment_ids for f in sent_features], dtype=torch.long
        ).cuda()
        return self.encoder.model.bert.embeddings(input_ids, segment_ids), input_mask

    def forward_with_sentences(
        self,
        sentences: List[List[str]],
        visual_feat: torch.Tensor,
        visual_boxes: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(sentences, (visual_feat, visual_boxes))

    def forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        visual_feat: torch.Tensor,
        visual_boxes: torch.Tensor,
    ) -> torch.Tensor:
        segment_ids = torch.zeros_like(mask).long()
        return self.encoder.model.bert(
            input_ids=ids,
            token_type_ids=segment_ids,
            attention_mask=mask,
            visual_feats=(visual_feat, visual_boxes),
            visual_attention_mask=None,
        )

    def forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        visual_feat: torch.Tensor,
        visual_boxes: torch.Tensor,
    ) -> torch.Tensor:
        segment_ids = torch.zeros_like(mask).long()
        return self.encoder.model.bert(
            input_ids=ids,
            token_type_ids=segment_ids,
            attention_mask=mask,
            visual_feats=(visual_feat, visual_boxes),
            visual_attention_mask=None,
        )

    def forward_with_question_attention(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        question_attention: torch.Tensor,
        visual_feat: torch.Tensor,
        visual_boxes: torch.Tensor,
    ) -> torch.Tensor:
        segment_ids = torch.zeros_like(mask).long()
        mask = (
            1.0
            - torch.cat(
                (
                    question_attention.unsqueeze(-1) * 2,
                    torch.ones_like(question_attention.unsqueeze(-1) * 2),
                ),
                dim=-1,
            ).min(-1)[0]
        ) * (-10000)
        return self.encoder.model.bert(
            input_ids=ids,
            token_type_ids=segment_ids,
            attention_mask=mask,
            visual_feats=(visual_feat, visual_boxes),
            visual_attention_mask=None,
        )

    def get_input_dim(self) -> int:
        return self.encoder.dim

    def get_output_dim(self) -> int:
        return self.get_input_dim()

    def is_bidirectional(self) -> bool:
        return False
