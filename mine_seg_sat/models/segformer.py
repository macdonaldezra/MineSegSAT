import math

import torch
import torch.nn.functional as F
from timm.models.layers import drop_path, trunc_normal_


class EfficientSelfAttention(torch.nn.Module):
    """
    A multi-head self-attention layer that improves computational efficiency by reducing the
    length of the input sequence to the attention layer.
    """

    def __init__(self, hidden_size, num_heads, dropout_p, sequence_reduction_ratio):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_heads})"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.attention_head_size = self.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.attention_head_size
        self.attn_score_divisor = self.attention_head_size**0.5

        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = torch.nn.Dropout(dropout_p)
        self.dense = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = torch.nn.Conv2d(
                hidden_size,
                hidden_size,
                kernel_size=sequence_reduction_ratio,
                stride=sequence_reduction_ratio,
            )
            self.layer_norm = torch.nn.LayerNorm(hidden_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        hidden_states,
        height,
        width,
        return_attn: bool = False,
    ):
        batch, n, c = hidden_states.shape
        q = (
            self.query(hidden_states)
            .reshape(batch, n, self.num_heads, c // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            # Reshape to (batch_size, channels, height, width)
            x = hidden_states.permute(0, 2, 1).reshape(batch, c, height, width)
            # Apply sequence reduction
            x = self.sr(x).reshape(batch, c, -1).permute(0, 2, 1)
            x = self.layer_norm(x)
            hidden_states = x

        k = (
            self.key(hidden_states)
            .reshape(batch, -1, self.num_heads, c // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.value(hidden_states)
            .reshape(batch, -1, self.num_heads, c // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.attn_score_divisor

        # Normalize the attention scores to probabilities
        attention_probs = torch.nn.functional.softmax(attn, dim=-1)
        attention_probs = self.dropout(attention_probs)

        outputs = torch.matmul(attention_probs, v).transpose(1, 2).reshape(batch, n, c)

        outputs = self.dense(outputs)
        outputs = self.dropout(outputs)
        if return_attn:
            attn_output = {
                "key": k,
                "value": v,
                "query": q,
                "attention": attention_probs,
            }
            return outputs, attn_output

        return outputs


class OverlapPatchEmbedding(torch.nn.Module):
    def __init__(self, patch_size: int, stride: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = torch.nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size // 2, patch_size // 2),
        )
        self.norm = torch.nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, height, width


class MixedFeedForwardNetwork(torch.nn.Module):
    def __init__(
        self, in_features, out_features, hidden_features, dropout_p: float = 0.0
    ):
        super().__init__()
        self.mlp_in = torch.nn.Linear(in_features, hidden_features)
        self.mlp_out = torch.nn.Linear(hidden_features, out_features)
        self.conv = torch.nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=True,
            groups=hidden_features,
        )
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, x, height, width):
        x = self.mlp_in(x)

        batch_size, _, num_channels = x.shape
        x = x.transpose(1, 2).reshape(batch_size, num_channels, height, width)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.nn.functional.gelu(x)

        x = self.dropout(x)
        x = self.mlp_out(x)
        x = self.dropout(x)

        return x


class TransformerBlock(torch.nn.Module):
    def __init__(
        self, input_dim, num_heads, dropout_p, dropout_path_p, sr_ratio
    ) -> None:
        super().__init__()
        self.attention = EfficientSelfAttention(
            hidden_size=input_dim,
            num_heads=num_heads,
            dropout_p=dropout_p,
            sequence_reduction_ratio=sr_ratio,
        )
        self.feed_forward = MixedFeedForwardNetwork(
            input_dim, input_dim, hidden_features=input_dim * 4, dropout_p=dropout_p
        )
        self.norm1 = torch.nn.LayerNorm(input_dim, eps=1e-6)
        self.norm2 = torch.nn.LayerNorm(input_dim, eps=1e-6)
        self.dropout_path_p = dropout_path_p

    def forward(self, x, height: int, width: int, return_attn: bool = False):
        # Normalize and compute Self-Attention
        skip = x
        x = self.norm1(x)
        if return_attn:
            x, attn_output = self.attention(x, height, width, return_attn)
        else:
            x = self.attention(x, height, width, return_attn)
        x = drop_path(x, drop_prob=self.dropout_path_p, training=self.training)
        x = x + skip

        # Normalize and compute Feed-Forward
        skip = x
        x = self.norm2(x)
        x = self.feed_forward(x, height, width)
        x = drop_path(x, drop_prob=self.dropout_path_p, training=self.training)
        x = x + skip
        if return_attn:
            return x, attn_output

        return x


class SegformerEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: tuple[int],
        embed_dims: tuple[int],
        num_heads: tuple[int],
        patch_sizes: tuple[int],
        strides: tuple[int],
        depths: tuple[int],
        sr_ratios: tuple[int],
        dropout_p: float,
        dropout_path_p: float,
    ) -> None:
        """
        The encoder for the SegFormer model including the patch embedding, transformer blocks,
        and the normalization.
        """
        super().__init__()

        # Initialize patch embedding layers
        embeddings = []
        for i in range(len(depths)):
            embeddings.append(
                OverlapPatchEmbedding(
                    patch_size=patch_sizes[i],
                    stride=strides[i],
                    in_channels=in_channels if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                )
            )

        self.embeddings = torch.nn.ModuleList(embeddings)

        # Initialize transformer blocks
        transformers = []
        pos = 0
        for i in range(len(depths)):
            blocks = []
            if i != 0:
                pos += depths[i - 1]

            # Initialize j transformer blocks for each depth
            for j in range(depths[i]):
                blocks.append(
                    TransformerBlock(
                        input_dim=embed_dims[i],
                        num_heads=num_heads[i],
                        dropout_p=dropout_p,
                        dropout_path_p=dropout_path_p * (pos + j) / (sum(depths) - 1),
                        sr_ratio=sr_ratios[i],
                    )
                )
            transformers.append(torch.nn.ModuleList(blocks))
            pos += depths[i]

        self.transformers = torch.nn.ModuleList(transformers)

        # Initialize layer norm for each depth
        layer_norm = []
        for i in range(len(depths)):
            layer_norm.append(torch.nn.LayerNorm(embed_dims[i], eps=1e-6))
        self.layer_norm = torch.nn.ModuleList(layer_norm)

    def forward(self, inputs, return_attn: bool = False):
        batch_size = inputs.shape[0]
        hidden_states = inputs
        outputs = []
        attn_outputs = []

        for embedding, transformer, norm in zip(
            self.embeddings, self.transformers, self.layer_norm
        ):
            if return_attn:
                stage_outputs = {}
                stage_outputs["patch_embedding_inputs"] = hidden_states

            # 1. Obtain patch embeddings
            hidden_states, height, width = embedding(hidden_states)
            #
            if return_attn:
                stage_outputs["patch_embedding_height"] = height
                stage_outputs["patch_embedding_width"] = width
                stage_outputs["patch_embedding_outputs"] = hidden_states

            # 2. Apply transformer blocks
            for block in transformer:
                if return_attn:
                    hidden_states, attn_output = block(
                        hidden_states, height, width, return_attn
                    )
                else:
                    hidden_states = block(hidden_states, height, width)

            # 3. Apply layer normalization
            hidden_states = norm(hidden_states)

            # 4. Reshape back to (b, c, h, w)
            hidden_states = hidden_states.reshape(
                batch_size, height, width, -1
            ).permute(0, 3, 1, 2)

            if return_attn:
                for k, v in attn_output.items():
                    stage_outputs[k] = v
                del attn_output
                attn_outputs.append(stage_outputs)

            # Append hidden states to all hidden states
            outputs.append(hidden_states)

        if return_attn:
            return outputs, attn_outputs

        return outputs

    def get_attention_outputs(self, inputs):
        """Returns the attention outputs of the encoder."""
        _, attn_outputs = self.forward(inputs, return_attn=True)

        return attn_outputs


class SegFormerDecoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: tuple[int],
        num_classes: int,
        embed_dim: int,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p

        # Initialize linear layers which unify the channel dimension of the encoder blocks
        # to the same as the fixed embedding dimension
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(channels, embed_dim, (1, 1))
                for channels in reversed(in_channels)
            ]
        )

        self.linear_fuse = torch.nn.Conv2d(
            embed_dim * len(self.in_channels), embed_dim, kernel_size=1, bias=False
        )
        self.batch_norm = torch.nn.BatchNorm2d(embed_dim, eps=1e-5)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.classifier = torch.nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        torch.nn.init.kaiming_normal_(
            self.linear_fuse.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.constant_(self.batch_norm.weight, 1)
        torch.nn.init.constant_(self.batch_norm.bias, 0)

    def forward(self, x):
        feature_size = x[0].shape[2:]

        x = [layer(x_i) for layer, x_i in zip(self.layers, reversed(x))]
        x = [
            F.interpolate(x_i, size=feature_size, mode="bilinear", align_corners=False)
            for x_i in x[:-1]
        ] + [x[-1]]

        x = self.linear_fuse(torch.cat(x[::-1], dim=1))
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


class SegFormer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_state_channels: tuple[int] = (64, 128, 320, 512),
        transformer_depth: tuple[int] = (3, 4, 18, 3),
        dropout_p: float = 0.1,
        dropout_path_p: float = 0.15,
        return_logits: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = SegformerEncoder(
            in_channels=in_channels,
            embed_dims=hidden_state_channels,
            num_heads=(1, 2, 5, 8),
            patch_sizes=(7, 3, 3, 3),
            strides=(4, 2, 2, 2),
            depths=transformer_depth,
            sr_ratios=(8, 4, 2, 1),
            dropout_p=dropout_p,
            dropout_path_p=dropout_path_p,
        )
        self.decoder = SegFormerDecoder(
            in_channels=hidden_state_channels,
            num_classes=num_classes,
            embed_dim=256,
            dropout_p=dropout_p,
        )
        self.return_logits = return_logits

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        image_dim = x.shape[2:]
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.nn.functional.interpolate(
            x, size=image_dim, mode="bilinear", align_corners=False
        )
        if not self.return_logits:
            x = torch.nn.functional.softmax(x, dim=1)

        return x

    def get_attention_outputs(self, x):
        attention_outputs = self.encoder.get_attention_outputs(x)
        return attention_outputs

    def get_last_attention_outputs(self, x):
        attention_outputs = self.encoder.get_attention_outputs(x)
        return attention_outputs[-1].get("attention", None)
