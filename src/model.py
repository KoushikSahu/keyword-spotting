import torch
import torch.nn as nn
import torch.nn.functional as F
from tsai.all import TCN
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf


class TCNModel(nn.Module):
  def __init__(self, n_classes, n_filters):
    super(TCNModel, self).__init__()
    self.tcn = TCN(c_in=39, c_out=n_classes, layers=[25]*n_filters)

  def forward(self, inp):
    return self.tcn(inp)


class DNNModel(nn.Module):
  def __init__(self, n_classes):
    super(DNNModel, self).__init__()
    self.n_classes = n_classes
    self.lin1 = nn.Linear(in_features=39*101, out_features=512)
    self.lin2 = nn.Linear(in_features=512, out_features=n_classes)

  def forward(self, inp):
    inp = torch.flatten(inp, start_dim=1)
    inp = F.relu(self.lin1(inp))
    out = self.lin2(inp)

    return out


def first_conv(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        # nn.PReLU()
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # (batchsize, channels_per_group, groups, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)  # (batchsize, -1, height, width)

    return x


def Base_block(oup_inc, stride):

    banch = nn.Sequential(
        # pw
        nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup_inc),
        nn.ReLU(inplace=True),
        # dw
        nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
        nn.BatchNorm2d(oup_inc),
        # pw-linear
        nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup_inc),
        nn.ReLU(inplace=True),
    )
    return banch


def EdgeCRNN_block(inp, oup_inc, stride):
    left_banch = nn.Sequential(
        # dw
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        # pw-linear
        nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup_inc),
        nn.ReLU(inplace=True),
    )
    right_banch = nn.Sequential(
        # pw
        nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup_inc),
        nn.ReLU(inplace=True),
        # dw
        nn.Conv2d(oup_inc, oup_inc, 3, stride,
                  1, groups=oup_inc, bias=False),
        nn.BatchNorm2d(oup_inc),
        # pw-linear
        nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup_inc),
        nn.ReLU(inplace=True),
    )
    return left_banch, right_banch


class EdgeCRNN_Residual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(EdgeCRNN_Residual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = Base_block(oup_inc, stride)
        else:
            self.banch1, self.banch2 = EdgeCRNN_block(inp, oup_inc, stride)

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = torch.chunk(x, 2, 1)[0]
            x2 = torch.chunk(x, 2, 1)[1]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class EdgeCRNN(nn.Module):
    def __init__(self, n_class=12, input_size=101, width_mult=1.):
        super(EdgeCRNN, self).__init__()

        # assert input_size % 32 == 0

        self.stage_repeats = [2, 3, 2]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            # *2  *2  16,  32,  64, 128, 256
            self.stage_out_channels = [-1, 16, 32, 64, 128, 256]
        elif width_mult == 1.0:
            # *4.9 *2  24, 72, 144, 288, 512
            self.stage_out_channels = [-1, 24, 72, 144, 288, 512]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]  # *7.3 *2
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 160, 320, 640, 1024]  # *9.3  *2
        else:
            raise ValueError(
                """groups is not supported for
                       1x1 Grouped Convolutions""")
        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = first_conv(1, input_channel)  # 1 dim
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.features = []
        # building Stage2-4
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    # inp, oup, stride, benchmodel):
                    self.features.append(EdgeCRNN_Residual(
                        input_channel, output_channel, 2, 2))
                else:
                    self.features.append(EdgeCRNN_Residual(
                        input_channel, output_channel, 1, 1))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)  # 16层网络
        # building last several layers
        self.conv_last = conv_1x1_bn(
            input_channel, self.stage_out_channels[-1])

        self.globalpool = nn.Sequential(nn.AvgPool2d(
            (3, 1), stride=(1, 1)))  # rnn->cnn (3,1)->(3, 7)
        # first-layer(3,1),other(2,1)； cnn first（3,7），other（2,4）

        # add RNN block
        self.hidden_size = 64
        # self.RNN = nn.RNN(self.stage_out_channels[-1], self.hidden_size, num_layers=1, batch_first=True)
        self.RNN = nn.LSTM(
            self.stage_out_channels[-1], self.hidden_size, num_layers=1, batch_first=True)
        # self.RNN = nn.GRU(self.stage_out_channels[-1], self.hidden_size, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(self.hidden_size, n_class))

        # building classifier CNN
        # self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))

    def forward(self, x):
        x = x.view(-1, 1, 39, 101)
        # print(x.shape)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        # print(x.shape)
        x = self.globalpool(x)  # shape(64,1024,1,4)

        # CNN
        # x = x.squeeze()
        # x = x.view(-1, self.stage_out_channels[-1])

        # add RNN block
        # shape(64,1024,1,4)--> shape(b, w, c)  (64, 7, 1024)
        x = x.squeeze(dim=2).permute(0, 2, 1)
        self.RNN.flatten_parameters()
        x, _ = self.RNN(x)  # shape(64, 7, 1024)
        x = x.permute(0, 2, 1).mean(2)  # shape(1, 64,1024)--> (64,1024, 7)

        x = self.classifier(x)
        return x


class SerializableModule(nn.Module):
  def __init__(self):
    super().__init__()

  def save(self, filename):
    torch.save(self.state_dict(), filename)

  def load(self, filename):
    self.load_state_dict(torch.load(
        filename, map_location=lambda storage, loc: storage))


class LSTM(SerializableModule):
  def __init__(self, n_labels):
    super().__init__()
    self.lstm = nn.LSTM(
        input_size=101,
        hidden_size=128,
        num_layers=2,
        batch_first=True,
    )
    self.linear = nn.Linear(128, n_labels, bias=False)

  def forward(self, x):
    embedding, (h_n, h_c) = self.lstm(x, None)
    y = self.linear(embedding[:, -1, :])
    # return y, embedding
    return y


class DS_Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DS_Convolution, self).__init__()
        self.dw_block = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.pw_block = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.dw_block(x)
        y = self.pw_block(y)
        return y


class DSCNN(SerializableModule):
  def __init__(self, n_labels=10):
    super(DSCNN, self).__init__()
    self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(25, 5), padding=(12, 2)),
                               nn.BatchNorm2d(64),
                               nn.ReLU())
    self.ds_block1 = DS_Convolution(
        in_channels=64, out_channels=64, kernel_size=(25, 5), padding=(12, 2))
    self.ds_block2 = DS_Convolution(
        in_channels=64, out_channels=64, kernel_size=(25, 5), padding=(12, 2))
    self.ds_block3 = DS_Convolution(
        in_channels=64, out_channels=64, kernel_size=(25, 5), padding=(12, 2))
    self.ds_block4 = DS_Convolution(
        in_channels=64, out_channels=64, kernel_size=(25, 5), padding=(12, 2))
    self.avg_pool = nn.AdaptiveAvgPool2d(1)  # global average pooling
    self.fc1 = nn.Linear(in_features=64, out_features=n_labels)

  def forward(self, x):
    x = x.unsqueeze(1)
    y = self.conv1(x)
    y = self.ds_block1(y)
    y = self.ds_block2(y)
    y = self.ds_block3(y)
    y = self.ds_block4(y)
    embedding = self.avg_pool(y)
    embedding = embedding.squeeze(-1).squeeze(-1)
    y = self.fc1(embedding)
    return y


def torch_to_tflite(model, filename, quantized=False):
    # pytorch model to onnx
    ONNX_PATH = f'models/{filename}.onnx'
    dummy_input = torch.randn(16, 39, 101).to('cuda')
    torch.onnx.export(
        model = model,
        args = dummy_input,
        f = ONNX_PATH,
        verbose = False,
        opset_version = 12, 
        input_names=['input'],
        output_names=['output']
    )

    # loading the saved onnx model
    onnx_model = onnx.load(ONNX_PATH)

    # convert with onnx-tf
    tf_rep = prepare(onnx_model)

    # exporting tf model
    TF_PATH = f'models/{filename}_tf'
    tf_rep.export_graph(TF_PATH)

    # tf to tflite
    TFLITE_PATH = f'models/{filename}.tflite'
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
    converter.experimental_enable_resource_variables = True
    # converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(TF_PATH)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tf_lite_model = converter.convert()
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tf_lite_model)

