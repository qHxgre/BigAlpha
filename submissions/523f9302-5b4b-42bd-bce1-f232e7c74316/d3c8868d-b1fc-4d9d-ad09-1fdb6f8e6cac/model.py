import torch
import torch.nn as nn


class AlphaTransformer(nn.Module):

    def __init__(
        self,
        feature_dim=10,
        d_model=32,
        nhead=2,
        num_layers=1
    ):

        super().__init__()


        self.embedding = nn.Linear(
            feature_dim,
            d_model
        )


        encoder_layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=nhead,

            batch_first=True,

            dropout=0.1

        )


        self.encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers

        )


        self.predictor = nn.Sequential(

            nn.Linear(
                d_model,
                16
            ),

            nn.ReLU(),

            nn.Linear(
                16,
                1
            )

        )


    def forward(self,x):

        x=self.embedding(x)

        x=self.encoder(x)


        # 时间维度平均
        x=x.mean(dim=1)


        out=self.predictor(x)


        return out.squeeze(-1)
