import dai
import torch
import pandas as pd


from model import AlphaTransformer



def main(

    data_source,

    start_date,

    end_date

):


    df=dai.query(

    f"""

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



    FROM {data_source}


    """,


    filters={

    "date":[

    start_date,

    end_date

    ]

    },


    compression=True


    ).df()



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



    model=AlphaTransformer()



    model.load_state_dict(

        torch.load(

            "alpha_transformer.pt",

            map_location="cpu"

        )

    )


    model.eval()



    results=[]



    for code,group in df.groupby(
        "instrument"
    ):


        group=group.sort_values(
            "date"
        )


        if len(group)<30:

            continue



        x=group[features].values[-30:]



        x=torch.tensor(

            x,

            dtype=torch.float32

        )


        x=x.unsqueeze(0)



        with torch.no_grad():

            score=model(x).item()



        results.append(

            [

            group["date"].iloc[-1],

            code,

            score

            ]

        )



    return pd.DataFrame(

        results,

        columns=[

        "date",

        "instrument",

        "score"

        ]

    )
