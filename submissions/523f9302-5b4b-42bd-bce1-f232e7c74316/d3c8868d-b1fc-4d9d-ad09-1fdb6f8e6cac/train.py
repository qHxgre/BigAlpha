import dai
import torch
import numpy as np


from torch.utils.data import Dataset,DataLoader

from sklearn.preprocessing import StandardScaler


from model import AlphaTransformer



print("train.py开始运行")



# ==========================
# 读取数据
# ==========================


print("开始读取数据")


df=dai.query(

"""

SELECT

date,
instrument,

open,
high,
low,
close,

volume,
amount,

bid_price1,
ask_price1,

bid_volume1,
ask_volume1


FROM bigalpha_2026_stock_bar1m


""",


filters={

"date":[

"2023-01-03",

"2023-01-04"

]

},


compression=True


).df()



print(
"原始数据量:",
df.shape
)




# ==========================
# 限制股票数量
# ==========================


codes=df.instrument.unique()[:200]


df=df[
df.instrument.isin(codes)
]


print(
"限制股票后:",
df.shape
)



df=df.sort_values(

[
"instrument",
"date"
]

)



features=[


"open",
"high",
"low",
"close",

"volume",
"amount",

"bid_price1",
"ask_price1",

"bid_volume1",
"ask_volume1"

]



# ==========================
# 标准化
# ==========================


scaler=StandardScaler()


df[features]=scaler.fit_transform(

df[features]

)



# ==========================
# Dataset
# ==========================


class AlphaDataset(Dataset):


    def __init__(

        self,

        df,

        window=30

    ):


        self.samples=[]


        for code,group in df.groupby(
            "instrument"
        ):


            group=group.sort_values(
                "date"
            )


            x=group[features].values


            price=group["close"].values



            for i in range(

                window,

                len(x)-5

            ):


                seq=x[

                    i-window:i

                ]


                future_return=(

                    price[i+5]

                    -

                    price[i]

                ) / price[i]



                self.samples.append(

                    (

                    seq.astype(
                        np.float32
                    ),

                    np.float32(
                        future_return
                    )

                    )

                )



    def __len__(self):

        return len(self.samples)



    def __getitem__(self,index):


        x,y=self.samples[index]


        return (

            torch.tensor(x),

            torch.tensor(y)

        )





dataset=AlphaDataset(df)



print(

"训练样本:",

len(dataset)

)



loader=DataLoader(

    dataset,

    batch_size=256,

    shuffle=True,

    num_workers=0

)



# ==========================
# 模型训练
# ==========================



device = (

"cuda"

if torch.cuda.is_available()

else

"cpu"

)



print(

"训练设备:",

device

)



model=AlphaTransformer()


model=model.to(device)



optimizer=torch.optim.AdamW(

    model.parameters(),

    lr=0.001

)


loss_fn=torch.nn.MSELoss()



epochs=3



for epoch in range(epochs):


    total_loss=0

    count=0



    for batch,(x,y) in enumerate(loader):


        x=x.to(device)

        y=y.to(device)



        pred=model(x)



        loss=loss_fn(

            pred,

            y

        )



        optimizer.zero_grad()



        loss.backward()



        optimizer.step()



        total_loss+=loss.item()

        count+=1



        if batch%20==0:


            print(

                "epoch",

                epoch,

                "batch",

                batch,

                "loss",

                loss.item()

            )




    print(

        "Epoch",

        epoch,

        "平均loss",

        total_loss/count

    )




# ==========================
# 保存模型
# ==========================


torch.save(

    model.state_dict(),

    "alpha_transformer.pt"

)



print(

"模型保存完成"

)
