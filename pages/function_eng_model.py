def eng_model_predict_2_5(days):
    num_days = days # วันที่ต้องการทำนาย

    import pycaret
    import pandas as pd

    # อ่านไฟล์ CSV
    data = pd.read_csv('export_data/filtered_data_3_best.csv')
    
    # ตั้ง 'timestamp' เป็น index หากยังไม่ได้ทำ
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    
    # ตั้งค่า freq ให้กับ index
    data = data.asfreq('D')  # 'D' สำหรับรายวัน
    
    # เติมค่าที่หายไปด้วยการ interpolate แบบเส้นตรง
    data.interpolate(method='linear', inplace=True)
    
    # กำหนดจำนวนวันที่ต้องการเพิ่ม
    num_days = num_days
    
    # สร้างวันที่ใหม่
    next_date = data.index[-1] + pd.Timedelta(days=1)
    new_dates = pd.date_range(start=next_date, periods=num_days, freq='D')
    
    # สร้างแถวใหม่
    new_rows = pd.DataFrame({'value': range(40, 40 + num_days)}, index=new_dates)
    
    # รวมข้อมูล (ป้องกัน index ซ้ำซ้อน)
    data = pd.concat([data, new_rows]).sort_index()
    
    # ลบ index ที่ซ้ำกัน (ถ้ามี)
    data = data.loc[~data.index.duplicated(keep='first')]

    last_date_temp = data.index[-1].strftime('%Y-%m-%d')
    data = data[['temperature']]
    data["year"] = data.index.year
    data["month"] = data.index.month
    data["day"] = data.index.day
    data["day_of_week"] = data.index.dayofweek  # 0 = Monday, 6 = Sunday
    data["week_of_year"] = data.index.isocalendar().week  # ค่าจะเป็น int64
    
    # สมมติว่า data มี index เป็น datetime อยู่แล้ว
    def assign_season(month):
        if month in [2, 3, 4]:  # ก.พ. - เม.ย.
            return "summer"        # ฤดูร้อน
        elif month in [5, 6, 7]:  # พ.ค. - ก.ค.
            return "early_rainy"   # ต้นฤดูฝน
        else:  # ส.ค. - ม.ค.
            return "late_rainy"    # ปลายฤดูฝน
    
    season_mapping = {'summer': 0, 'early_rainy': 1, 'late_rainy': 2}
    data["season"] = data.index.month.map(assign_season)
    data['season'] = data['season'].map(season_mapping)
    
    data["is_weekend"] = data.index.dayofweek.isin([5, 6]).astype(int)
    
    import pandas as pd
    import numpy as np
    
    # สมมุติว่ามี DataFrame data ที่มีข้อมูลอยู่แล้ว
    # เพิ่มฟีเจอร์ day_of_year
    data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
    data['day_of_year'] = data['date'].dt.dayofyear
    
    # เพิ่มฟีเจอร์ month_sin และ month_cos
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    
    # เพิ่มฟีเจอร์ day_of_week_sin และ day_of_week_cos
    data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    
    lag_size = 7
    for i in range(1,lag_size+1):
        data[f'temperature_lag_{i}'] = data['temperature'].shift(i)
    
    data_filtered = data.loc['2023-09':]  # ดึงข้อมูลตั้งแต่กันยายน 2023 เป็นต้นไป
    # สมมติว่า data มีจำนวนวันทั้งหมด
    total_data = len(data_filtered)
    
    # จำนวนวันที่ต้องการเป็น test data
    test_days = 30
    
    # คำนวณสัดส่วนของ 30 วันในข้อมูลทั้งหมด
    test_size = test_days / total_data
    
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data_filtered, test_size=test_size, shuffle=False)
    
    
    from pycaret.regression import load_model, predict_model
    
    model_temp = load_model("main/temp_model_1")
    data_compare = data_filtered.loc[:'2025-01-12']
    result = pd.DataFrame()
    
    start_date = "2025-01-13"
    end_date = last_date_temp
    
    model = model_temp
    
    for date in pd.date_range(start=start_date, end=end_date, freq="D"):
        # Drop columns
        test_1 = data_filtered
        # ดึงวันนี้มา
        test_1 = test_1.loc[str(date.date())]
    
        # ใส่เข้าไป
        data_compare = pd.concat([data_compare, test_1], axis=0)
    
        for i in range(1,lag_size+1):
            data_compare[f'temperature_lag_{i}'] = data_compare['temperature'].shift(i)
    
        # ดึง
        test_1 = (data_compare.loc[str(date.date())]).drop(columns=['temperature'])
        
        # Drop ทิ้งล่าสุดใน data_compare
        data_compare = data_compare.drop(data_compare.last_valid_index())
        test_1 = test_1[~test_1.index.duplicated(keep='last')]
    
        # Predict
        predict_test_1 = predict_model(model, data=test_1)
    
        # เพิ่มผลทำนายเข้าไปใน data_compare
        data_compare = pd.concat([data_compare, predict_test_1], axis=0)
    
        # แทนที่ค่าในคอลัมน์ 'pm_2_5' ในวันที่ '2025-01-13' ด้วยค่าในคอลัมน์ 'prediction_label'
        data_compare.loc[str(date.date()), 'temperature'] = data_compare.loc[str(date.date()), 'prediction_label']
    
        # ลบคอลัมน์ 'prediction_label' (หากต้องการแค่ผลทำนาย)
        data_compare = data_compare.drop(columns=['prediction_label'])
    
        # กรองเอาวันที่ซ้ำกัน (index ซ้ำ) ออกให้เหลือแค่ตัวเดียว
        data_compare = data_compare[~data_compare.index.duplicated(keep='last')]
    
        # บันทึกลง result
        result = pd.concat([result, predict_test_1], axis=0)
    
    #-----------PLOT GRAPH TEMP---------
    real_data = data_filtered.loc[start_date:end_date]
    # สมมุติว่า result คือ DataFrame ที่รวม predictions กับ pm_10
    result_plot = pd.concat([result['prediction_label'].shift(-1), real_data['temperature']], axis=1)
    result_plot.columns = ['Predictions', 'Actual TEMP']  # ตั้งชื่อคอลัมน์
    #-----------------------------------

    #โหลด humi model
    model_humi = load_model("main/humi_use_temp_model_2")
    
    # อ่านไฟล์ CSV
    data_humi = pd.read_csv('export_data/filtered_data_3_best.csv')

    # ลบแถวล่าสุด (แถวสุดท้าย)
    data_humi = data_humi.drop(data_humi.index[-1])
    # ตั้ง 'timestamp' เป็น index หากยังไม่ได้ทำ
    data_humi['timestamp'] = pd.to_datetime(data_humi['timestamp'])
    data_humi.set_index('timestamp', inplace=True)
    
    # ตั้งค่า freq ให้กับ index
    data_humi = data_humi.asfreq('D')  # 'D' สำหรับรายวัน
    
    # เติมค่าที่หายไปด้วยการ interpolate แบบเส้นตรง
    data_humi.interpolate(method='linear', inplace=True)  

    # กำหนดจำนวนวันที่ต้องการเพิ่ม
    num_days = num_days
    
    # สร้างวันที่ใหม่
    next_date = data_humi.index[-1] + pd.Timedelta(days=1)
    new_dates = pd.date_range(start=next_date, periods=num_days, freq='D')
    
    # สร้างแถวใหม่
    new_rows = pd.DataFrame({'value': range(40, 40 + num_days)}, index=new_dates)
    
    # รวมข้อมูล (ป้องกัน index ซ้ำซ้อน)
    data_humi = pd.concat([data_humi, new_rows]).sort_index()
    
    # ลบ index ที่ซ้ำกัน (ถ้ามี)
    data_humi = data_humi.loc[~data_humi.index.duplicated(keep='first')]

    last_date_humi = data_humi.index[-1].strftime('%Y-%m-%d')
    
    # add temp predict column
    shifted_predictions_temp = result['prediction_label'].shift(-1).iloc[:-1]
    
    data_humi.loc[shifted_predictions_temp.index, 'temperature'] = shifted_predictions_temp

    data_humi = data_humi[['humidity','temperature']]
    data_humi["year"] = data_humi.index.year
    data_humi["month"] = data_humi.index.month
    data_humi["day"] = data_humi.index.day
    data_humi["day_of_week"] = data_humi.index.dayofweek  # 0 = Monday, 6 = Sunday
    data_humi["week_of_year"] = data_humi.index.isocalendar().week  # ค่าจะเป็น int64
    
    # สมมติว่า data_humi มี index เป็น datetime อยู่แล้ว
    def assign_season(month):
        if month in [2, 3, 4]:  # ก.พ. - เม.ย.
            return "summer"        # ฤดูร้อน
        elif month in [5, 6, 7]:  # พ.ค. - ก.ค.
            return "early_rainy"   # ต้นฤดูฝน
        else:  # ส.ค. - ม.ค.
            return "late_rainy"    # ปลายฤดูฝน
    
    season_mapping = {'summer': 0, 'early_rainy': 1, 'late_rainy': 2}
    data_humi["season"] = data_humi.index.month.map(assign_season)
    data_humi['season'] = data_humi['season'].map(season_mapping)
    
    data_humi["is_weekend"] = data_humi.index.dayofweek.isin([5, 6]).astype(int)
    
    import pandas as pd
    import numpy as np
    
    # สมมุติว่ามี data_humiFrame data_humi ที่มีข้อมูลอยู่แล้ว
    # เพิ่มฟีเจอร์ day_of_year
    data_humi['date'] = pd.to_datetime(data_humi[['year', 'month', 'day']])
    data_humi['day_of_year'] = data_humi['date'].dt.dayofyear
    
    lag_size = 7
    for i in range(1,lag_size+1):
        data_humi[f'humidity_lag_{i}'] = data_humi['humidity'].shift(i)
    
    for i in range(1,lag_size+1):
        data_humi[f'temperature_lag_{i}'] = data_humi['temperature'].shift(i)
    data_filtered = data_humi.loc['2023-09':]  # ดึงข้อมูลตั้งแต่กันยายน 2023 เป็นต้นไป
    data_compare = data_filtered.loc[:'2025-01-12']
    result_humi = pd.DataFrame()
    
    start_date = "2025-01-13"
    end_date = last_date_humi
    
    model = model_humi
    
    for date in pd.date_range(start=start_date, end=end_date, freq="D"):
        # Drop columns
        test_1 = data_filtered
        # ดึงวันนี้มา
        test_1 = test_1.loc[str(date.date())]
    
        # ใส่เข้าไป
        data_compare = pd.concat([data_compare, test_1], axis=0)
    
        for i in range(1,lag_size+1):
            data_compare[f'humidity_lag_{i}'] = data_compare['humidity'].shift(i)
    
        # ดึง
        test_1 = (data_compare.loc[str(date.date())]).drop(columns=['humidity'])
        
        # Drop ทิ้งล่าสุดใน data_compare
        data_compare = data_compare.drop(data_compare.last_valid_index())
        test_1 = test_1[~test_1.index.duplicated(keep='last')]
    
        # Predict
        predict_test_1 = predict_model(model, data=test_1)
    
        # เพิ่มผลทำนายเข้าไปใน data_compare
        data_compare = pd.concat([data_compare, predict_test_1], axis=0)
    
        # แทนที่ค่าในคอลัมน์ 'pm_2_5' ในวันที่ '2025-01-13' ด้วยค่าในคอลัมน์ 'prediction_label'
        data_compare.loc[str(date.date()), 'humidity'] = data_compare.loc[str(date.date()), 'prediction_label']
    
        # ลบคอลัมน์ 'prediction_label' (หากต้องการแค่ผลทำนาย)
        data_compare = data_compare.drop(columns=['prediction_label'])
    
        # กรองเอาวันที่ซ้ำกัน (index ซ้ำ) ออกให้เหลือแค่ตัวเดียว
        data_compare = data_compare[~data_compare.index.duplicated(keep='last')]
    
        # บันทึกลง result_humi
        result_humi = pd.concat([result_humi, predict_test_1], axis=0)
    
    #-----------PLOT GRAPH HUMI---------
    real_data = data_filtered.loc[start_date:end_date]
    # สมมุติว่า result คือ DataFrame ที่รวม predictions กับ pm_10
    result_plot = pd.concat([result_humi['prediction_label'].shift(-1), real_data['humidity']], axis=1)
    result_plot.columns = ['Predictions', 'Actual HUMI']  # ตั้งชื่อคอลัมน์
    #------------------------------------

    #โหลดโมเดล PM 2.5
    model_pm_2_5 = load_model("main/pm2_5_model_1")

    # อ่านไฟล์ CSV
    data_pm_2_5 = pd.read_csv('export_data/filtered_data_3_best.csv')
    # ลบแถวล่าสุด (แถวสุดท้าย)
    data_pm_2_5 = data_pm_2_5.drop(data_pm_2_5.index[-1])
    # ตั้ง 'timestamp' เป็น index หากยังไม่ได้ทำ
    data_pm_2_5['timestamp'] = pd.to_datetime(data_pm_2_5['timestamp'])
    data_pm_2_5.set_index('timestamp', inplace=True)
    
    # ตั้งค่า freq ให้กับ index
    data_pm_2_5 = data_pm_2_5.asfreq('D')  # 'D' สำหรับรายวัน
    
    # เติมค่าที่หายไปด้วยการ interpolate แบบเส้นตรง
    data_pm_2_5.interpolate(method='linear', inplace=True)

    # กำหนดจำนวนวันที่ต้องการเพิ่ม
    num_days = num_days
    
    # สร้างวันที่ใหม่
    next_date = data_pm_2_5.index[-1] + pd.Timedelta(days=1)
    new_dates = pd.date_range(start=next_date, periods=num_days, freq='D')
    
    # สร้างแถวใหม่
    new_rows = pd.DataFrame({'value': range(40, 40 + num_days)}, index=new_dates)
    
    # รวมข้อมูล (ป้องกัน index ซ้ำซ้อน)
    data_pm_2_5 = pd.concat([data_pm_2_5, new_rows]).sort_index()
    
    # ลบ index ที่ซ้ำกัน (ถ้ามี)
    data_pm_2_5 = data_pm_2_5.loc[~data_pm_2_5.index.duplicated(keep='first')]

    last_date_pm_2_5 = data_humi.index[-1].strftime('%Y-%m-%d')

    # add humi predict column
    # เลื่อนค่าของ prediction_label ไปข้างหน้า 1 วัน และตัดค่าแถวสุดท้ายที่ไม่มีผล
    shifted_predictions = result_humi['prediction_label'].shift(-1).iloc[:-1]
    
    # สมมติว่า data_pm_2_5 และ shifted_predictions มีอยู่แล้ว
    # data_pm_2_5 เป็น DataFrame ที่มีคอลัมน์ 'humidity' และ index เป็น timestamp
    # shifted_predictions เป็น Series ที่มี index เป็น timestamp ในช่วงวันที่ 2025-01-13 ถึง 2025-02-10
    
    # แทนที่ค่าของคอลัมน์ 'humidity' ใน data_pm_2_5 ด้วยค่าจาก shifted_predictions ในวันที่ตรงกัน
    data_pm_2_5.loc[shifted_predictions.index, 'humidity'] = shifted_predictions
    
    # add temp predict column
    shifted_predictions_temp = result['prediction_label'].shift(-1).iloc[:-1]
    
    data_pm_2_5.loc[shifted_predictions_temp.index, 'temperature'] = shifted_predictions_temp

    data_pm_2_5["year"] = data_pm_2_5.index.year
    data_pm_2_5["month"] = data_pm_2_5.index.month
    data_pm_2_5["day"] = data_pm_2_5.index.day
    data_pm_2_5["day_of_week"] = data_pm_2_5.index.dayofweek  # 0 = Monday, 6 = Sunday
    data_pm_2_5["week_of_year"] = data_pm_2_5.index.isocalendar().week  # ค่าจะเป็น int64
    
    # สมมติว่า data_pm_2_5 มี index เป็น datetime อยู่แล้ว
    def assign_season(month):
        if month in [2, 3, 4]:  # ก.พ. - เม.ย.
            return "summer"        # ฤดูร้อน
        elif month in [5, 6, 7]:  # พ.ค. - ก.ค.
            return "early_rainy"   # ต้นฤดูฝน
        else:  # ส.ค. - ม.ค.
            return "late_rainy"    # ปลายฤดูฝน
    
    season_mapping = {'summer': 0, 'early_rainy': 1, 'late_rainy': 2}
    data_pm_2_5["season"] = data_pm_2_5.index.month.map(assign_season)
    data_pm_2_5['season'] = data_pm_2_5['season'].map(season_mapping)
    
    data_pm_2_5["is_weekend"] = data_pm_2_5.index.dayofweek.isin([5, 6]).astype(int)
    data_pm_2_5['dew_point'] = data_pm_2_5['temperature'] - ((100 - data_pm_2_5['humidity']) / 5)
    # อัตราส่วนของอุณหภูมิต่อความชื้น
    data_pm_2_5["temp_humidity_ratio"] = data_pm_2_5["temperature"] / (data_pm_2_5["humidity"] + 1)
    # ดัชนีความร้อน (Heat Index) หรืออุณหภูมิที่รับรู้
    T = data_pm_2_5["temperature"]
    RH = data_pm_2_5["humidity"]
    
    data_pm_2_5["heat_index"] = -42.379 + (2.04901523 * T) + (10.14333127 * RH) - (0.22475541 * T * RH) - \
                       (0.00683783 * T**2) - (0.05481717 * RH**2) + (0.00122874 * T**2 * RH) + \
                       (0.00085282 * T * RH**2) - (0.00000199 * T**2 * RH**2)
    
    # ความชื้นสัมพัทธ์เชิงดัชนี
    data_pm_2_5["rh_index"] = data_pm_2_5["humidity"] ** 2 / (data_pm_2_5["temperature"] + 1)
    
    # เพิ่มฟีเจอร์ day_of_year
    data_pm_2_5['date'] = pd.to_datetime(data_pm_2_5[['year', 'month', 'day']])
    data_pm_2_5['day_of_year'] = data_pm_2_5['date'].dt.dayofyear
    
    # เพิ่มฟีเจอร์ month_sin และ month_cos
    data_pm_2_5['month_sin'] = np.sin(2 * np.pi * data_pm_2_5['month'] / 12)
    data_pm_2_5['month_cos'] = np.cos(2 * np.pi * data_pm_2_5['month'] / 12)
    
    # เพิ่มฟีเจอร์ day_of_week_sin และ day_of_week_cos
    data_pm_2_5['day_of_week_sin'] = np.sin(2 * np.pi * data_pm_2_5['day_of_week'] / 7)
    data_pm_2_5['day_of_week_cos'] = np.cos(2 * np.pi * data_pm_2_5['day_of_week'] / 7)
    
    lag_size = 30
    for i in range(1,lag_size+1):
        data_pm_2_5[f'pm_2_5_lag_{i}'] = data_pm_2_5['pm_2_5'].shift(i)
    
    data_pm_2_5['humidity_lag_1'] = data_pm_2_5['humidity'].shift(1)
    data_pm_2_5['humidity_lag_2'] = data_pm_2_5['humidity'].shift(2)
    data_pm_2_5['humidity_lag_3'] = data_pm_2_5['humidity'].shift(3)
    data_pm_2_5['humidity_lag_7'] = data_pm_2_5['humidity'].shift(7)
    
    data_pm_2_5['temperature_lag_1'] = data_pm_2_5['temperature'].shift(1)
    data_pm_2_5['temperature_lag_2'] = data_pm_2_5['temperature'].shift(2)
    data_pm_2_5['temperature_lag_3'] = data_pm_2_5['temperature'].shift(3)
    data_pm_2_5['temperature_lag_7'] = data_pm_2_5['temperature'].shift(7)
    data_filtered = data_pm_2_5.loc['2023-09':]  # ดึงข้อมูลตั้งแต่กันยายน 2023 เป็นต้นไป
    data_compare = data_filtered.loc[:'2025-01-12']
    result_pm_2_5 = pd.DataFrame()
    
    start_date = "2025-01-13"
    end_date = last_date_pm_2_5
    
    model = model_pm_2_5
    
    for date in pd.date_range(start=start_date, end=end_date, freq="D"):
        # Drop columns
        test_1 = data_filtered
        
        # ดึงวันนี้มา
        test_1 = test_1.loc[str(date.date())]
    
        # ใส่เข้าไป
        data_compare = pd.concat([data_compare, test_1], axis=0)
    
        for i in range(1,lag_size+1):
            data_compare[f'pm_2_5_lag_{i}'] = data_compare['pm_2_5'].shift(i)
    
        # ดึง
        test_1 = (data_compare.loc[str(date.date())]).drop(columns=['pm_2_5'])
        
        # Drop ทิ้งล่าสุดใน data_compare
        data_compare = data_compare.drop(data_compare.last_valid_index())
        test_1 = test_1[~test_1.index.duplicated(keep='last')]
    
        # Predict
        predict_test_1 = predict_model(model, data=test_1)
    
        # เพิ่มผลทำนายเข้าไปใน data_compare
        data_compare = pd.concat([data_compare, predict_test_1], axis=0)
    
        # แทนที่ค่าในคอลัมน์ 'pm_2_5' ในวันที่ '2025-01-13' ด้วยค่าในคอลัมน์ 'prediction_label'
        data_compare.loc[str(date.date()), 'pm_2_5'] = data_compare.loc[str(date.date()), 'prediction_label']
    
        # ลบคอลัมน์ 'prediction_label' (หากต้องการแค่ผลทำนาย)
        data_compare = data_compare.drop(columns=['prediction_label'])
    
        # กรองเอาวันที่ซ้ำกัน (index ซ้ำ) ออกให้เหลือแค่ตัวเดียว
        data_compare = data_compare[~data_compare.index.duplicated(keep='last')]
    
        # บันทึกลง result_pm_2_5
        result_pm_2_5 = pd.concat([result_pm_2_5, predict_test_1], axis=0)
    
    real_data = data_filtered.loc[start_date:end_date]
    # สมมุติว่า result คือ DataFrame ที่รวม predictions กับ pm_10
    result_plot_2_5 = pd.concat([result_pm_2_5['prediction_label'].shift(-1), real_data['pm_2_5']], axis=1)
    result_plot_2_5.columns = ['Predictions', 'Actual PM2.5']  # ตั้งชื่อคอลัมน์

    return  result_plot_2_5