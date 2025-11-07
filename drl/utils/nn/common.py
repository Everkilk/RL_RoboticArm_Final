import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

#################################################################################################################
####################################### LINEAR AND DYNAMICS POOLING LAYERS ######################################
#################################################################################################################
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Linear(nn.Sequential):
    def __init__(self, in_features, out_features, norm_layer='none', activation='ReLU', dropout=0.0):
        assert norm_layer in ['none', 'spectral', 'layer', 'rms'], ValueError
        modules = OrderedDict([])
        # weighted layer
        modules['linear'] = nn.Linear(in_features=in_features, out_features=out_features, bias=(norm_layer != 'layer'))
        # normalization layer
        if norm_layer == 'spectral':
            modules['linear'] = nn.utils.parametrizations.spectral_norm(modules['linear'])
        elif norm_layer == 'layer':
            modules['norm'] = nn.LayerNorm(normalized_shape=out_features)
        elif norm_layer == 'rms':
            modules['norm'] = RMSNorm(normalized_shape=out_features)
        # activation layer
        if activation is not None:
            assert isinstance(activation, (str, nn.Module)), TypeError
            if isinstance(activation, str):
                activation = eval(f'nn.{activation}()')
            modules['act'] = activation
        # dropout layer
        if dropout > 1e-3:
            modules['drop'] = nn.Dropout(p=dropout)
        
        super().__init__(modules)


class SAGlobalPool(nn.Module):
    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()
        self.attn = nn.Linear(in_features=embed_dim, out_features=1)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        """
        Input:
            x: input query embedding features, shape=[batch_size, num_anchors, emb_dim]
        Output:
            ap_x: average weighted output, shape=[batch_size, emb_dim]
        """
        attn = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * self.dropout(attn), dim=1) # [batch_size, emb_dim]


###############################################################################################################
########################################## CONVOLUTION LAYERS #################################################
###############################################################################################################
class Conv1D(nn.Sequential):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding='same',
                 batch_norm=False, dropout=0.0):
        modules = OrderedDict([])
        if padding == 'same':
            padding = kernel_size // 2
        modules['conv'] = nn.Conv1d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=not batch_norm)
        if batch_norm:
            modules['bn'] = nn.BatchNorm1d(num_features=out_channels)
        if dropout > 1e-3:
            modules['drop'] = nn.Dropout(p=dropout)
        super().__init__(modules)

    def fuse(self):
        conv = self._modules.pop('conv')
        bn = self._modules.pop('bn', None)
        self._modules.pop('drop', None)
        if bn is not None:
            in_channels = conv.in_channels
            out_channels = conv.out_channels
            fused_conv = conv.__class__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=conv.kernel_size,
                                        stride=conv.stride,
                                        dilation=conv.dilation,
                                        padding=conv.padding,
                                        bias=True).to(conv.weight.device)
            with torch.no_grad():
                weight = ((bn.weight / torch.sqrt(bn.eps + bn.running_var)).view(out_channels, -1) * conv.weight.view(out_channels, -1))
                weight = weight.view(fused_conv.weight.shape)
                fused_conv.weight.copy_(weight)
                if conv.bias is not None:
                    b_conv = conv.bias
                else:
                    b_conv = torch.zeros_like(bn.weight)
                bias = ((bn.weight * (b_conv - bn.running_mean) / torch.sqrt(bn.eps + bn.running_var)) + bn.bias)
                bias = bias.view(fused_conv.bias.shape)
                fused_conv.bias.copy_(bias)
            conv = fused_conv
        return conv


class Conv2D(nn.Sequential):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding='same',
                 batch_norm=False, dropout=0.0):
        modules = OrderedDict([])
        if padding == 'same':
            padding = kernel_size // 2
        modules['conv'] = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=not batch_norm)
        if batch_norm:
            modules['bn'] = nn.BatchNorm2d(num_features=out_channels)
        if dropout > 1e-3:
            modules['drop'] = nn.Dropout(p=dropout)
        super().__init__(modules)

    def fuse(self):
        conv = self._modules.pop('conv')
        bn = self._modules.pop('bn', None)
        self._modules.pop('drop', None)
        if bn is not None:
            in_channels = conv.in_channels
            out_channels = conv.out_channels
            fused_conv = conv.__class__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=conv.kernel_size,
                                        stride=conv.stride,
                                        dilation=conv.dilation,
                                        padding=conv.padding,
                                        bias=True).to(conv.weight.device)
            with torch.no_grad():
                weight = ((bn.weight / torch.sqrt(bn.eps + bn.running_var)).view(out_channels, -1) * conv.weight.view(out_channels, -1))
                weight = weight.view(fused_conv.weight.shape)
                fused_conv.weight.copy_(weight)
                if conv.bias is not None:
                    b_conv = conv.bias
                else:
                    b_conv = torch.zeros_like(bn.weight)
                bias = ((bn.weight * (b_conv - bn.running_mean) / torch.sqrt(bn.eps + bn.running_var)) + bn.bias)
                bias = bias.view(fused_conv.bias.shape)
                fused_conv.bias.copy_(bias)
            conv = fused_conv
        return conv


class RecurrentConv(nn.Module):
    def __init__(self, channels, activation, num_iters=2):
        super().__init__()
        self.conv = Conv2D(in_channels=channels, out_channels=channels, kernel_size=3)
        self.act = activation
        self.num_iters = num_iters

    def forward(self, x):
        xt = self.act(self.conv(x))
        for _ in range(self.num_iters - 1):
            xt = self.act(self.conv(x + xt))
        return xt

    def fuse(self):
        self.conv = self.conv.fuse()
        return self


class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, dropout=0.0):
        super().__init__()
        self.conv = Conv2D(in_channels=in_channels, 
                           out_channels=out_channels, 
                           kernel_size=3, 
                           batch_norm=batch_norm,
                           dropout=dropout)
        self.conv_1x1 = Conv2D(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=1,
                               batch_norm=batch_norm,
                               dropout=dropout)
        if (in_channels == out_channels) and batch_norm:
            self.bn_identity = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.bn_identity = None
    
    def forward(self, x):
        if self.bn_identity is None:
            return self.conv(x) + self.conv_1x1(x)
        return self.conv(x) + self.conv_1x1(x) + self.bn_identity(x)

    def fuse(self):
        conv = self.conv.fuse()
        conv_1x1 = self.conv_1x1.fuse()

        conv.weight.data += torch.nn.functional.pad(conv_1x1.weight.data, [1, 1, 1, 1])
        conv.bias.data += conv_1x1.bias.data
        if self.bn_identity is not None:
            channels, device = self.bn_identity.num_features, conv.weight.device
            conv_identity = Conv2D(in_channels=channels, out_channels=channels, kernel_size=1).to(device)
            conv_identity.conv.weight.data = torch.eye(channels, device=device).unsqueeze(-1).unsqueeze(-1)
            conv_identity.bn = self.bn_identity
            conv_identity = conv_identity.fuse()

            conv.weight.data += torch.nn.functional.pad(conv_identity.weight.data, [1, 1, 1, 1])
            conv.bias.data += conv_identity.bias.data
        
        return conv


###################################################################################################################
########################################## RNN ARCHITECTURES ######################################################
###################################################################################################################
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        # weighted layers
        self.i2h = nn.Linear(in_features=input_size, out_features=2 * hidden_size, bias=bias)
        self.h2h = nn.Linear(in_features=hidden_size, out_features=2 * hidden_size, bias=bias)
        self.lin_in = nn.Linear(in_features=input_size, out_features=hidden_size, bias=bias)
        self.lin_hp = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=bias)

    def forward(self, x, h, m=None):
        if m is not None:
            h = h * m.view(-1, 1)
        # linear mapping
        zr = torch.sigmoid(self.i2h(x) + self.h2h(h))
        z, r = torch.chunk(input=zr, chunks=2, dim=1)
        n = torch.tanh(self.lin_in(x) + r * self.lin_hp(h))
        return (1 - z) * h + z * n


class LayerNormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        # normalization layers
        self.ln_i2h = nn.LayerNorm(normalized_shape=2 * hidden_size, elementwise_affine=False)
        self.ln_h2h = nn.LayerNorm(normalized_shape=2 * hidden_size, elementwise_affine=False)
        self.ln_in = nn.LayerNorm(normalized_shape=hidden_size, elementwise_affine=False)
        self.ln_hp = nn.LayerNorm(normalized_shape=hidden_size, elementwise_affine=False)

        # weighted layers
        self.i2h = nn.Linear(in_features=input_size, out_features=2 * hidden_size, bias=bias)
        self.h2h = nn.Linear(in_features=hidden_size, out_features=2 * hidden_size, bias=bias)
        self.lin_in = nn.Linear(in_features=input_size, out_features=hidden_size, bias=bias)
        self.lin_hp = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=bias)

    def forward(self, x, h, m=None):
        if m is not None:
            h = h * m.view(-1, 1)
        # linear mapping
        zr = torch.sigmoid(self.ln_i2h(self.i2h(x)) + self.ln_h2h(self.h2h(h)))
        z, r = torch.chunk(input=zr, chunks=2, dim=1)
        n = torch.tanh(self.ln_in(self.lin_in(x)) + r * self.ln_hp(self.lin_hp(h)))
        return (1 - z) * h + z * n
    

class SeqGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, use_norm=False, dropout=0.0, bias=True):
        super().__init__()
        cell = LayerNormGRUCell if use_norm else GRUCell
        self.cells = nn.ModuleList([])
        for _ in range(num_layers):
            self.cells.append(cell(input_size=input_size, hidden_size=hidden_size, bias=bias))
            input_size = hidden_size
        self.ln = nn.LayerNorm(normalized_shape=hidden_size) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(p=dropout) if dropout > 1e-3 else nn.Identity()
        self.register_buffer('h0', torch.zeros(1, hidden_size))

    def forward(self, seq, mask=None):
        # seq_shape = (batch_size, time_steps, input_size)
        # mask_shape = (batch_size, time_steps)
        b, t, _ = seq.size()
        if mask is not None:
            if mask.dtype == torch.bool:
                mask = mask.float()
            assert mask.size() == (b, t), ValueError(f'mask shape must be ({b}, {t})!!!')
        else:
            mask = torch.ones(b, t, device=self.h0.device)
        h = len(self.cells) * [self.h0.repeat(b, 1)]
        for i in range(t):
            x, m = seq[:, i], mask[:, i]
            for l, cell in enumerate(self.cells):
                x = cell(x, h[l], m)
                h[l] = x
        return self.dropout(self.ln(x))


###################################################################################################################
########################################### TRANSFORMER BLOCK #####################################################
###################################################################################################################
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, head_dim=64, dropout=0.1, causal=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # query-key-value linear projection
        self.fc_q = nn.Linear(in_features=embed_dim, out_features=num_heads * head_dim, bias=False)
        self.fc_k = nn.Linear(in_features=embed_dim, out_features=num_heads * head_dim, bias=False)
        self.fc_v = nn.Linear(in_features=embed_dim, out_features=num_heads * head_dim, bias=False)

        # output linear projection
        self.fc_o = nn.Linear(in_features=num_heads * head_dim, out_features=embed_dim)

        # dropout layer
        self.dropout = nn.Dropout(p=dropout)
        self.causal = causal

    def forward(self, query, key, value, mask=None):
        """
        Compute attentive feature
        Input:
            query: a feature vector that describes what we are looking for in the sequence, shape=(batch_size, q_seq_len, embed_dim)
            key: a feature vector roughly describes what the element is "offering", or when it might be important, -
                                                                                    shape=(batch_size, kv_seq_len, embed_dim)
            value: a feature vector is the one we want to average over, shape=(batch_size, kv_seq_len, embed_dim)
            mask: is used specifically to prevent the model from "cheating" during training, shape=(batch_size, num_heads, q_seq_len, kv_seq_len)
        Output:
            output: attentive feature, shape=(batch_size, q_seq_len, embed_dim)
        """
        bs, nh, hd = query.size(0), self.num_heads, self.head_dim

        # query-key-value projections
        query = self.fc_q(query).view(bs, -1, nh, hd).transpose(1, 2) # (batch_size, num_heads, q_seq_len, embed_dim)
        key = self.fc_k(key).view(bs, -1, nh, hd).transpose(1, 2) # (batch_size, num_heads, kv_seq_len, embed_dim)
        value = self.fc_v(value).view(bs, -1, nh, hd).transpose(1, 2) # (batch_size, num_heads, kv_seq_len, embed_dim)

        # scaled-dot product attention
        energy = ((query @ key.transpose(-1, -2)) / (hd ** 0.5)) # (batch_size, num_heads, q_seq_len, kv_seq_len)
        if self.causal:
            future_mask = self.get_causal_mask(query.size(2), key.size(2), query.device)
            mask = future_mask if mask is None else (mask & future_mask)

        if mask is not None:
            energy = energy.masked_fill(~mask, -1e10)
        attn = torch.softmax(energy, dim=-1) # (batch_size, num_heads, q_seq_len, kv_seq_len)

        # compute attentive feature
        x = (self.dropout(attn) @ value).transpose(1, 2).reshape(bs, -1, nh * hd)  # (batch_size, q_seq_len, num_heads * head_dim)
        x = self.dropout(self.fc_o(x)) # (batch_size, q_seq_len, embed_dim)

        return x, attn
    
    @staticmethod
    def get_causal_mask(q_seq_len, kv_seq_len, device):
        return torch.ones(1, 1, q_seq_len, kv_seq_len, dtype=torch.bool, device=device).tril()

class PositionWiseFF(nn.Module): # position-wise feed forward
    def __init__(self, embed_dim, expand_dim=2, activation=nn.ReLU(), dropout=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(in_features=embed_dim, out_features=embed_dim * expand_dim)
        self.fc_2 = nn.Linear(in_features=embed_dim * expand_dim, out_features=embed_dim)
        assert isinstance(activation, nn.Module), TypeError
        self.act = activation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(self.act(self.fc_1(x)))
        return self.dropout(self.fc_2(x))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, head_dim=64, expand_dim=2,
                 activation=nn.ReLU(), dropout=0.1, norm_eps=1e-5):
        super().__init__()
        if activation is not None:
            assert isinstance(activation, (str, nn.Module)), TypeError
            if isinstance(activation, str):
                activation = eval(f'nn.{activation}()')

        # self-attention part
        self.self_attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout)
        self.self_attn_norm = nn.LayerNorm(normalized_shape=embed_dim, eps=norm_eps)

        # feed forward part
        self.ff = PositionWiseFF(embed_dim=embed_dim, expand_dim=expand_dim, activation=activation, dropout=dropout)
        self.ff_norm = nn.LayerNorm(normalized_shape=embed_dim, eps=norm_eps)

    def forward(self, embeds, attention_mask=None):
        # self-attention
        embeds = embeds + self.self_attn(embeds, embeds, embeds, attention_mask)[0]
        embeds = self.self_attn_norm(embeds)

        # feed forward
        return self.ff_norm(embeds + self.ff(embeds))
    
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_layers, embed_dim=256, num_heads=8, head_dim=None, expand_dim=2,
                 activation=nn.SiLU(), dropout=0.1, norm_eps=1e-5, max_length=128):
        super().__init__()
        head_dim = (embed_dim // num_heads) if head_dim is None else head_dim
        assert isinstance(max_length, int) and max_length > 0, ValueError
        self.max_length = max_length

        # embedding tables and parameters
        self.pos_emb = nn.Embedding(num_embeddings=max_length, embedding_dim=embed_dim)
        self.token_emb = nn.Linear(in_features=input_dim, out_features=embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('emb_scale', torch.tensor(head_dim ** 0.5).float())

        # transformer encoder block
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    head_dim=head_dim,
                                    expand_dim=expand_dim,
                                    activation=activation,
                                    norm_eps=norm_eps)
            for _ in range(num_layers)
        ])

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, device = x.size(0), x.size(1), x.device
        assert seq_len <= self.max_length, ValueError

        # compute embedding feature
        pos = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        embeds = self.dropout(self.token_emb(x) * self.emb_scale + self.pos_emb(pos))
        
        # feature extraction
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        for block in self.blocks:
            embeds = block(embeds, attention_mask)

        return embeds


#############################################################################################
#################################### KAN ACTIVATION #########################################
#############################################################################################
def b_spline(x, grid, spline_order=3):
    """
        B-Spline Function
        Input:
            x: input value, shape=(..., dim)
            grid: control points, shape=(grid_size,)
        Output:
            bases: output value, shape=(..., dim, grid_size)
    """
    x = x.unsqueeze(dim=-1)
    bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
    for k in range(1, spline_order + 1):
        bases = (
            (x - grid[:-(k+1)]) / (grid[k:-1] - grid[:-(k+1)]) * bases[..., :-1] 
            + 
            ((grid[(k+1):] - x) / (grid[(k+1):] - grid[1:-k])) * bases[..., 1:]
        )
    return bases


class BSpline(nn.Module):
    def __init__(self, grid_min=-2.0, grid_max=2.0, grid_size=8, spline_order=3):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.register_buffer(
            'grid', torch.linspace(grid_min, grid_max, grid_size + spline_order + 1)
        )
        
    def forward(self, x):
        return b_spline(x=x, grid=self.grid, spline_order=self.spline_order)


def radial_basis_function(x, grid, denominator=None):
    """
        Radial Basis Function
        Input:
            x: input value, shape=(..., dim)
            grid: center points, shape=(grid_size,)
            denominator: normalization parameter, float
        Output:
            bases: output value, shape=(..., dim, grid_size)
    """
    if denominator is None:
        denominator = (grid.max() - grid.min()) / (grid.size(1) - 1)
    return torch.exp(-((x[..., None] - grid) / denominator) ** 2)


class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min=-2.0, grid_max=2.0, grid_size=8, denominator=None):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.grid_size = grid_size
        self.denominator = denominator if denominator is not None else (grid_max - grid_min) / (grid_size - 1)
        self.register_buffer(
            'grid', torch.linspace(grid_min, grid_max, grid_size)
        )
        
    def forward(self, x):
        return radial_basis_function(x=x, grid=self.grid, denominator=self.denominator)


#############################################################################################
######################################### KAN LAYER #########################################
#############################################################################################
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=RadialBasisFunction(), use_norm=True, use_bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_norm = use_norm
        self.use_bias = use_bias
            
        # set activation function
        assert isinstance(activation, nn.Module), TypeError
        self.activation = activation
        
        # set normalization layer
        if self.use_norm:
            # self.norm = RMSNorm(normalized_shape=in_features)
            self.norm = nn.LayerNorm(normalized_shape=in_features)
        
        # intialize weights
        self.coef_weight = nn.Parameter(0.01 * torch.randn(out_features, in_features * self.activation.grid_size))
        if self.use_bias:
            self.base_weight = nn.Parameter(0.01 * torch.randn(out_features, in_features))
        
    def forward(self, x):
        # normalize input
        if self.use_norm:
            x = self.norm(x)
        # run forward
        out = F.linear(
            input=self.activation(x).view(*x.size()[:-1], -1), 
            weight=self.coef_weight
        )
        if self.use_bias:
            out = out + F.linear(F.silu(x), self.base_weight)
        return out