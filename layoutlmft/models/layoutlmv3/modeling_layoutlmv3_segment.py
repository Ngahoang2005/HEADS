# coding=utf-8
"""LayoutLMv3 + predicted-segment contextual fusion (V5).

Design goal
-----------
Keep the strong vanilla LayoutLMv3 token-classification path and use
AUTOMATICALLY PREDICTED text segments only as contextual information.

Pipeline:
    word-level bbox
        -> LayoutLMv3
        -> token hidden states h_i
        -> mean-pool hidden states inside each predicted segment
        -> lightweight segment self-attention
        -> broadcast contextual segment representation back to tokens
        -> gated residual fusion in hidden space
        -> SAME token classifier
        -> BIO logits

Important differences from the previous V4:
1. No gold/oracle segment IDs are required by this model.
2. No segment classifier and no direct logit correction.
3. No consensus/voting/uncertainty heuristic.
4. Segment information is fused BEFORE the token classifier.
5. The segment residual projection starts at zero, so the newly added branch
   starts as an exact no-op rather than immediately disturbing the baseline.
6. The model is fully end-to-end: gradients can flow through segment pooling,
   segment context, fusion, and the LayoutLMv3 backbone.

Expected runtime environment:
    Python 3.7
    torch 1.10
    transformers 4.16.2
"""

import math

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput

from .modeling_layoutlmv3 import (
    LayoutLMv3ClassificationHead,
    LayoutLMv3Model,
    LayoutLMv3PreTrainedModel,
)


class LayoutLMv3ForSegmentTokenClassification(LayoutLMv3PreTrainedModel):
    """LayoutLMv3 token classification with predicted-segment contextual fusion."""

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super(LayoutLMv3ForSegmentTokenClassification, self).__init__(config)

        self.num_labels = config.num_labels
        hidden = config.hidden_size

        # ------------------------------------------------------------------
        # 1) Original LayoutLMv3 path.
        # ------------------------------------------------------------------
        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.num_labels < 10:
            self.classifier = nn.Linear(hidden, config.num_labels)
        else:
            self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)

        # ------------------------------------------------------------------
        # 2) Segment context encoder.
        # ------------------------------------------------------------------
        ctx_layers = int(getattr(config, "segment_context_layers", 1))
        ctx_heads = int(getattr(config, "segment_context_heads", 4))
        ctx_dropout = float(
            getattr(config, "segment_context_dropout", config.hidden_dropout_prob)
        )
        max_seg_pos = int(getattr(config, "segment_context_max_positions", 128))

        if ctx_layers <= 0:
            raise ValueError("segment_context_layers must be > 0")
        if ctx_heads <= 0:
            raise ValueError("segment_context_heads must be > 0")
        if max_seg_pos <= 0:
            raise ValueError("segment_context_max_positions must be > 0")
        if hidden % ctx_heads != 0:
            raise ValueError(
                "segment_context_heads ({}) must divide hidden_size ({})".format(
                    ctx_heads, hidden
                )
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=ctx_heads,
            dim_feedforward=hidden * 2,
            dropout=ctx_dropout,
            batch_first=True,
        )
        self.segment_context = nn.TransformerEncoder(
            encoder_layer,
            num_layers=ctx_layers,
        )
        self.segment_position_embedding = nn.Embedding(max_seg_pos, hidden)

        # ------------------------------------------------------------------
        # 3) Hidden-space gated fusion.
        # ------------------------------------------------------------------
        # [token h, segment c, h*c, |h-c|]
        fusion_in = hidden * 4
        fusion_hidden = int(
            getattr(config, "segment_fusion_hidden", min(256, max(128, hidden // 2)))
        )

        self.segment_fusion_gate = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.GELU(),
            nn.Dropout(float(getattr(config, "segment_fusion_dropout", ctx_dropout))),
            nn.Linear(fusion_hidden, 1),
        )

        # Residual message maps segment context back to token hidden space.
        self.segment_residual = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # Start conservatively. The residual path is exactly zero at
        # initialization, so loading a vanilla LayoutLMv3 checkpoint produces
        # the same logits before the new branch learns anything.
        gate_bias = float(getattr(config, "segment_fusion_gate_bias", -2.0))

        # ------------------------------------------------------------------
        # 4) Training knobs.
        # ------------------------------------------------------------------
        self.segment_message_dropout = float(
            getattr(config, "segment_message_dropout", 0.0)
        )
        if not 0.0 <= self.segment_message_dropout < 1.0:
            raise ValueError("segment_message_dropout must be in [0, 1)")

        self._segment_debug = str(__import__("os").environ.get("SEGMENT_DEBUG", "0")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._segment_debug_done = False

        self.init_weights()

        with torch.no_grad():
            # Position embeddings can be learned normally.
            nn.init.normal_(self.segment_position_embedding.weight, mean=0.0, std=0.02)

            # Crucial stabilization: segment residual starts as zero.
            for module in self.segment_residual:
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)

            # Keep the gate conservative initially.
            for module in self.segment_fusion_gate:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(module.bias)
            nn.init.constant_(self.segment_fusion_gate[-1].bias, gate_bias)

    @staticmethod
    def _normalize_seg_id(seg_id, text_len):
        """Align a [B, L] segment-id tensor to the text sequence length."""
        if seg_id is None:
            return None
        if seg_id.ndim != 2:
            raise ValueError(
                "seg_id must be 2-D, got shape={}".format(tuple(seg_id.shape))
            )
        if seg_id.shape[1] == text_len:
            return seg_id
        if seg_id.shape[1] > text_len:
            return seg_id[:, :text_len]
        pad = torch.full(
            (seg_id.shape[0], text_len - seg_id.shape[1]),
            -1,
            device=seg_id.device,
            dtype=seg_id.dtype,
        )
        return torch.cat([seg_id, pad], dim=1)

    @staticmethod
    def _index_segments(seg_id):
        """Map arbitrary segment IDs to compact per-token indices.

        The model does not assume that raw predicted segment IDs are contiguous.
        Segment IDs only identify membership; their numeric values have no
        semantic meaning.
        """
        batch_size, seq_len = seg_id.shape
        valid = seg_id >= 0

        if not bool(valid.any()):
            return valid, None, None, None, None

        # Use a sufficiently large multiplier so IDs from different batch
        # elements can never collide.
        max_seg = int(seg_id[valid].max().item())
        multiplier = max_seg + 1
        batch_idx = (
            torch.arange(batch_size, device=seg_id.device)
            .unsqueeze(1)
            .expand(batch_size, seq_len)
        )
        keys = (batch_idx * multiplier + seg_id.clamp(min=0))[valid]

        unique_keys, inverse = torch.unique(
            keys,
            sorted=True,
            return_inverse=True,
        )
        seg_batch = torch.div(
            unique_keys,
            multiplier,
            rounding_mode="floor",
        )
        seg_counts = torch.bincount(
            seg_batch.cpu(),
            minlength=batch_size,
        ).to(seg_batch.device)
        starts = torch.cumsum(seg_counts, dim=0) - seg_counts
        seg_pos = (
            torch.arange(unique_keys.numel(), device=seg_id.device)
            - starts[seg_batch]
        )
        return valid, inverse, seg_batch, seg_pos, seg_counts

    def _segment_context(
        self,
        text_hidden,
        valid,
        inverse,
        seg_batch,
        seg_pos,
        seg_counts,
    ):
        """Mean-pool token states per segment and exchange context across segments."""
        batch_size, _, hidden = text_hidden.shape
        num_segments = int(seg_batch.numel())

        token_states = text_hidden[valid]
        token_states_float = token_states.float()

        # Mean pooling.
        sums = torch.zeros(
            num_segments,
            hidden,
            device=token_states.device,
            dtype=torch.float32,
        )
        sums.index_add_(0, inverse, token_states_float)

        counts = torch.bincount(
            inverse.cpu(),
            minlength=num_segments,
        ).to(inverse.device).clamp(min=1).unsqueeze(1).float()
        segment_mean = (sums / counts).to(text_hidden.dtype)

        # Pack segments per batch element.
        max_segments = max(int(seg_counts.max().item()), 1)
        packed = text_hidden.new_zeros(batch_size, max_segments, hidden)
        packed[seg_batch, seg_pos] = segment_mean

        # One mask row per document.
        pad_mask = (
            torch.arange(max_segments, device=token_states.device)
            .unsqueeze(0)
            >= seg_counts.unsqueeze(1)
        )
        # Transformer requires the first real position to remain unmasked.
        if max_segments > 0:
            pad_mask[:, 0] = False

        # Segment positions follow the compact order created from the raw IDs.
        positions = torch.arange(
            max_segments,
            device=token_states.device,
        ).clamp(max=self.segment_position_embedding.num_embeddings - 1)
        packed = packed + self.segment_position_embedding(positions).unsqueeze(0)

        context = self.segment_context(
            packed,
            src_key_padding_mask=pad_mask,
        )

        # Return contextual representation for every valid token.
        token_segment_context = context[seg_batch, seg_pos]
        return token_segment_context, segment_mean

    def _build_segment_fusion(self, text_hidden, seg_id):
        """Create a hidden-space residual from predicted segments."""
        seg_id = self._normalize_seg_id(seg_id, text_hidden.shape[1]).long()

        indexed = self._index_segments(seg_id)
        valid, inverse, seg_batch, seg_pos, seg_counts = indexed

        if not bool(valid.any()):
            return None

        segment_context, segment_mean = self._segment_context(
            text_hidden,
            valid,
            inverse,
            seg_batch,
            seg_pos,
            seg_counts,
        )

        h = text_hidden[valid]
        c = segment_context[inverse]

        fusion_features = torch.cat(
            [
                h,
                c,
                h * c,
                (h - c).abs(),
            ],
            dim=-1,
        )

        gate = torch.sigmoid(self.segment_fusion_gate(fusion_features))
        message = self.segment_residual(c)

        if self.training and self.segment_message_dropout > 0.0:
            # Drop complete segment messages, not individual hidden dimensions.
            num_segments = int(seg_counts.sum().item())
            keep = (
                torch.rand(
                    num_segments,
                    device=h.device,
                    dtype=message.dtype,
                )
                >= self.segment_message_dropout
            ).to(message.dtype)
            message = message * keep[inverse].unsqueeze(-1)

        correction_valid = gate * message

        correction = text_hidden.new_zeros(
            text_hidden.shape[0],
            text_hidden.shape[1],
            text_hidden.shape[2],
        )
        correction[valid] = correction_valid

        if self._segment_debug and not self._segment_debug_done:
            with torch.no_grad():
                print(
                    "[SEGMENT_V5_DEBUG] valid_tokens={} segments={} "
                    "token_hidden_norm={:.4f} segment_mean_norm={:.4f} "
                    "message_norm={:.4f} gate_mean={:.4f} correction_norm={:.4f}".format(
                        int(valid.sum().item()),
                        int(seg_batch.numel()),
                        h.float().norm(dim=-1).mean().item(),
                        segment_mean.float().norm(dim=-1).mean().item(),
                        message.float().norm(dim=-1).mean().item(),
                        gate.float().mean().item(),
                        correction.float().norm(dim=-1).mean().item(),
                    )
                )
            self._segment_debug_done = True

        return correction

    @staticmethod
    def _masked_ce(logits, labels, attention_mask, num_labels):
        loss_fct = CrossEntropyLoss()
        logits_flat = logits.reshape(-1, num_labels)
        labels_flat = labels.reshape(-1)

        if attention_mask is None:
            return loss_fct(logits_flat, labels_flat)

        active = attention_mask.reshape(-1) == 1
        n_active = active.numel()
        n_labels = labels_flat.numel()

        if n_labels < n_active:
            labels_flat = torch.cat(
                [
                    labels_flat,
                    torch.full(
                        (n_active - n_labels,),
                        loss_fct.ignore_index,
                        device=labels.device,
                        dtype=labels.dtype,
                    ),
                ],
                dim=0,
            )
        elif n_labels > n_active:
            labels_flat = labels_flat[:n_active]

        active_labels = torch.where(
            active,
            labels_flat,
            torch.full_like(labels_flat, loss_fct.ignore_index),
        )
        return loss_fct(logits_flat, active_labels)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        valid_span=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        seg_id=None,
        line_ids=None,
        block_ids=None,
        column_ids=None,
        entity_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        images=None,
    ):
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            valid_span=valid_span,
        )

        sequence_output = outputs[0]
        text_len = (
            input_ids.shape[1]
            if input_ids is not None
            else inputs_embeds.shape[1]
        )
        text_hidden = sequence_output[:, :text_len, :]
        image_hidden = sequence_output[:, text_len:, :]

        # --------------------------------------------------------------
        # Vanilla path first. If seg_id=None, this is the normal model.
        # --------------------------------------------------------------
        fused_text_hidden = text_hidden

        if seg_id is not None:
            segment_correction = self._build_segment_fusion(
                text_hidden,
                seg_id,
            )
            if segment_correction is not None:
                fused_text_hidden = text_hidden + segment_correction

        # Same classifier as vanilla LayoutLMv3, but on fused hidden states.
        logits_text = self.classifier(self.dropout(fused_text_hidden))

        if image_hidden.shape[1] > 0:
            image_logits = self.classifier(self.dropout(image_hidden))
            logits = torch.cat([logits_text, image_logits], dim=1)
        else:
            logits = logits_text

        loss = None
        if labels is not None:
            token_labels = labels[:, :text_len]
            token_attention = (
                attention_mask[:, :text_len]
                if attention_mask is not None
                else None
            )
            loss = self._masked_ce(
                logits_text,
                token_labels,
                token_attention,
                self.num_labels,
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
