from network_base import NetworkBase
from pytorch_pretrained_bert import BertModel as PytorchBERT
from torch import nn



class UnidirectionalLSTM(NetworkBase):

    def __init__(self, device):

        super(UnidirectionalLSTM, self).__init__(device=device)

        self.lstm = nn.LSTM(input_size =300, hidden_size =150)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=42000, out_features=3)
        self.softmax = nn.Softmax(dim=1)

        self.to(self.device)


    def forward(self, x):

        x, _ = self.lstm(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x



class BidirectionalLSTM(NetworkBase):

    def __init__(self, device):

        super(BidirectionalLSTM, self).__init__(device=device)

        self.lstm = nn.LSTM(input_size =300, hidden_size =150, bidirectional=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=84000, out_features=3) # 84000 = 150 * 250
        self.softmax = nn.Softmax(dim=1)

        self.to(self.device)


    def forward(self, x):

        x, _ = self.lstm(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x



class PyramidalLSTM(NetworkBase):

    def __init__(self, device):

        super(PyramidalLSTM, self).__init__(device=device)

        self.lstm1 = nn.LSTM(input_size =300, hidden_size =64, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size =128, hidden_size =32, bidirectional=False)
        self.lstm3 = nn.LSTM(input_size =64, hidden_size =16, bidirectional=False)
        self.lstm4 = nn.LSTM(input_size =32, hidden_size =8, bidirectional=False)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=280, out_features=3)
        self.softmax = nn.Softmax(dim=1)

        self.to(self.device)


    def forward(self, x):

        x, _ = self.lstm1(x)
        x = x.view([-1, 140, 128])
        x, _ = self.lstm2(x)
        x = x.view([-1, 70, 64])
        x, _ = self.lstm3(x)
        x = x.view([-1, 35, 32])
        x, _ = self.lstm4(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x 



class BertModel(NetworkBase):

    def __init__(self, device):

        super(BertModel, self).__init__(device=device)

        self.bert = PytorchBERT.from_pretrained('bert-base-uncased')
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=768, out_features=3)
        self.softmax = nn.Softmax(dim=1)

        self.to(self.device)


    def forward(self, x):
        
        x = self.bert(x)[1]
        x = self.linear(x)
        x = self.softmax(x)
        return x 