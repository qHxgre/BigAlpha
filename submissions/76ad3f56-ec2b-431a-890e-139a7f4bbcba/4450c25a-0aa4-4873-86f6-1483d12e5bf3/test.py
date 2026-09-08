import dai

df = dai.query('''select brand_or_origin from au_spot_latest_price group by brand_or_origin''')

print(df)
