# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pooling functions to aggregate frame-level deep features
into segment-level speaker embeddings

High-order statistics are surprisingly effective, TSDP acts similarly as TSTP,
even though we remove the mean statistic, on Voxceleb.
"""
import mlx.core as mx
import mlx.nn as nn


class TAP(nn.Module):
    """
    Temporal average pooling, only first-order mean is considered
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TAP, self).__init__()
        self.in_dim = in_dim

    def __call__(self, x):
        pooling_mean = mx.mean(x, axis=-1)
        # To be compatable with 2D input
        pooling_mean = pooling_mean.flatten(start_axis=1)
        return pooling_mean

    def get_out_dim(self):
        self.out_dim = self.in_dim
        return self.out_dim


class TSDP(nn.Module):
    """
    Temporal standard deviation pooling, only second-order std is considered
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSDP, self).__init__()
        self.in_dim = in_dim

    def __call__(self, x):
        # The last dimension is the temporal axis
        pooling_std = mx.sqrt(mx.var(x, axis=-1) + 1e-7)
        pooling_std = pooling_std.flatten(start_axis=1)
        return pooling_std

    def get_out_dim(self):
        self.out_dim = self.in_dim
        return self.out_dim


class TSTP(nn.Module):
    """
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSTP, self).__init__()
        self.in_dim = in_dim

    def __call__(self, x):
        # The last dimension is the temporal axis
        pooling_mean = mx.mean(x, axis=-1)
        pooling_std = mx.sqrt(mx.var(x, axis=-1) + 1e-7)
        pooling_mean = pooling_mean.flatten(start_axis=1)
        pooling_std = pooling_std.flatten(start_axis=1)
        stats = mx.concatenate((pooling_mean, pooling_std), axis=1)
        return stats

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2
        return self.out_dim


class ASTP(nn.Module):
    """Attentive statistics pooling: Channel- and context-dependent
    statistics pooling, first used in ECAPA_TDNN.
    """

    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False, **kwargs):
        super(ASTP, self).__init__()
        self.in_dim = in_dim
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, bottleneck_dim, kernel_size=1
            )  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, bottleneck_dim, kernel_size=1
            )  # equals W and b in the paper
        self.linear2 = nn.Conv1d(
            bottleneck_dim, in_dim, kernel_size=1
        )  # equals V and k in the paper

    def __call__(self, x):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = mx.mean(x, axis=-1)[:, :, None]
            context_mean = mx.broadcast_to(context_mean, x.shape)
            context_std = mx.sqrt(mx.var(x, axis=-1) + 1e-7)[:, :, None]
            context_std = mx.broadcast_to(context_std, x.shape)
            x_in = mx.concatenate((x, context_mean, context_std), axis=1)
        else:
            x_in = x

        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = mx.tanh(
            self.linear1(x_in.transpose(0, 2, 1)).transpose(0, 2, 1)
        )  # alpha = F.relu(self.linear1(x_in))
        alpha = mx.softmax(
            self.linear2(alpha.transpose(0, 2, 1)).transpose(0, 2, 1), axis=2
        )
        mean = mx.sum(alpha * x, axis=2)
        var = mx.sum(alpha * (x**2), axis=2) - mean**2
        std = mx.sqrt(mx.clip(var, 1e-7, None))
        return mx.concatenate([mean, std], axis=1)

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim


class MHASTP(nn.Module):
    """Multi head attentive statistics pooling
    Reference:
        Self Multi-Head Attention for Speaker Recognition
        https://arxiv.org/pdf/1906.09890.pdf
    """

    def __init__(
        self, in_dim, layer_num=2, head_num=2, d_s=1, bottleneck_dim=64, **kwargs
    ):
        super(MHASTP, self).__init__()
        assert (
            in_dim % head_num
        ) == 0  # make sure that head num can be divided by input_dim
        self.in_dim = in_dim
        self.head_num = head_num
        d_model = int(in_dim / head_num)
        channel_dims = [bottleneck_dim for i in range(layer_num + 1)]
        if d_s > 1:
            d_s = d_model
        else:
            d_s = 1
        self.d_s = d_s
        channel_dims[0], channel_dims[-1] = d_model, d_s
        self.heads_att_trans = []
        for i in range(self.head_num):
            layers = []
            for j in range(layer_num - 1):
                layers.extend(
                    [
                        nn.Conv1d(channel_dims[j], channel_dims[j + 1], 1, 1),
                        nn.Tanh(),
                    ]
                )
            layers.append(
                nn.Conv1d(channel_dims[layer_num - 1], channel_dims[layer_num], 1, 1)
            )
            self.heads_att_trans.append(nn.Sequential(*layers))

    def __call__(self, input):
        """
        input: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(input.shape) == 4:  # B x F x T
            input = input.reshape(
                input.shape[0], input.shape[1] * input.shape[2], input.shape[3]
            )
        assert len(input.shape) == 3
        bs, f_dim, t_dim = input.shape
        chunks = mx.split(input, self.head_num, axis=1)
        # split
        chunks_out = []
        for i, layer in enumerate(self.heads_att_trans):
            att_score = layer(chunks[i].transpose(0, 2, 1)).transpose(0, 2, 1)
            alpha = mx.softmax(att_score, axis=-1)
            mean = mx.sum(alpha * chunks[i], axis=2)
            var = mx.sum(alpha * chunks[i] ** 2, axis=2) - mean**2
            std = mx.sqrt(mx.clip(var, 1e-7, None))
            chunks_out.append(mx.concatenate((mean, std), axis=1))
        out = mx.concatenate(chunks_out, axis=1)
        return out

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim


class MQMHASTP(nn.Module):
    """An attentive pooling
    Reference:
        multi query multi head attentive statistics pooling
        https://arxiv.org/pdf/2110.05042.pdf
    Args:
        in_dim: the feature dimension of input
        layer_num: the number of layer in the pooling layer
        query_num: the number of querys
        head_num: the number of heads
        bottleneck_dim: the bottleneck dimension

    SA (H = 1, Q = 1, n = 2, d_s = 1) ref:
        https://www.danielpovey.com/files/2018_interspeech_xvector_attention.pdf
    MHA (H > 1, Q = 1, n = 1, d_s = 1) ref:
        https://arxiv.org/pdf/1906.09890.pdf
    AS (H = 1, Q > 1, n = 2, d_s = 1) ref:
        https://arxiv.org/pdf/1803.10963.pdf
    VSA (H = 1, Q > 1, n = 2, d_s = d_h) ref:
        http://www.interspeech2020.org/uploadfile/pdf/Mon-2-10-5.pdf
    """

    def __init__(
        self,
        in_dim,
        layer_num=2,
        query_num=2,
        head_num=8,
        d_s=2,
        bottleneck_dim=64,
        **kwargs
    ):
        super(MQMHASTP, self).__init__()
        self.n_query = [
            MHASTP(
                in_dim,
                layer_num=layer_num,
                head_num=head_num,
                d_s=d_s,
                bottleneck_dim=bottleneck_dim,
            )
            for i in range(query_num)
        ]
        self.query_num = query_num
        self.in_dim = in_dim

    def __call__(self, input):
        """
        input: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(input.shape) == 4:  # B x F x T
            input = input.reshape(
                input.shape[0], input.shape[1] * input.shape[2], input.shape[3]
            )
        assert len(input.shape) == 3
        res = []
        for i, layer in enumerate(self.n_query):
            res.append(layer(input))
        out = mx.concatenate(res, axis=-1)
        return out

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2 * self.query_num
        return self.out_dim


if __name__ == "__main__":
    data = mx.random.normal(shape=(16, 512, 10, 35))
    # model = StatisticsPooling()
    model = MQMHASTP(512 * 10)
    model = MHASTP(512 * 10)
    model = MQMHASTP(512 * 10, context=False)
    print(model)

    out = model(data)
    print(out.shape)
    print(model.get_out_dim())
