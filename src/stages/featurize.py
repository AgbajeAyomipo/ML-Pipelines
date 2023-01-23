import pandas as pd
import yaml
import os

os.chdir('C:/Users/Ayo Agbaje/Desktop/ML-Pipelines/notebooks')
def featurize() -> None:

    with open('../params.yaml') as config_:
        config__ = yaml.safe_load(config_)

    df_ = pd.read_csv(config__['data_load']['dataset_csv'])
    df_['PRICE'] = df_['Price']
    df_ = df_.drop('Price', axis = 1)
    df_ = df_.sort_values('PRICE', ascending = False).reset_index().drop('index', axis = 1)
    df_ = df_.iloc[3:, :]
    
    def levy_(annot_):
        if annot_ == '-':
            return 0
        else:
            return int(annot_)

    df_['Levy'] = df_['Levy'].apply(levy_)

    def engine_volume(annot_):
        if ' ' in annot_:
            return float(annot_.split(' ')[0])
        else:
            return float(annot_)

    df_['Engine volume'] = df_['Engine volume'].apply(engine_volume) 

    def mileage_(annot_):
        return int(annot_.split(' ')[0])

    df_['Mileage'] = df_['Mileage'].apply(engine_volume)   

    category_map = {'Coupe': 0,'Jeep': 1,'Sedan': 2,'Universal': 3,'Pickup': 4,'Minivan': 5,'Goods wagon': 6,
                    'Microbus': 7,'Cabriolet': 8,'Hatchback': 9,'Limousine': 10}
    doors_map = {'02-Mar': 0,
                '04-May': 1,
                '>5': 2}
    fuel_map = {'Petrol': 0,'Diesel': 1,'Hybrid': 2,'Plug-in Hybrid': 3,'LPG': 4,'CNG': 5,'Hydrogen': 6}
    wheel_map = {'Left wheel': 0,
                'Right-hand drive': 1}

    leather_interior_map = {'Yes': 0,
                            'No': 1}

    drive_wheels_map = {'Rear': 0,
                        '4x4': 1,
                        'Front': 2}

    gear_box_map = {'Automatic': 0,
                    'Tiptronic': 1,
                    'Manual': 2,
                    'Variator': 3}

    manufacturer_map = {'PORSCHE': 0,'LAND ROVER': 2,'MERCEDES-BENZ': 3,'BMW': 4,'LEXUS': 5,'BENTLEY': 6,'TOYOTA': 7,
                        'JAGUAR': 8,'JEEP': 9,'FORD': 10,'HYUNDAI': 11,'HONDA': 12,'MITSUBISHI': 13,'AUDI': 14,'CHEVROLET': 15,
                        'OPEL': 16,'FERRARI': 17,'KIA': 18,'PEUGEOT': 19,'SUZUKI': 20,'VOLKSWAGEN': 21,'MAZDA': 22,'HUMMER': 23,
                        'SSANGYONG':  24,'ASTON MARTIN': 25,'TESLA': 26,'GAZ': 27,'MINI': 28,'CADILLAC': 29,'NISSAN': 30,
                        'SKODA': 31,'ACURA': 32,'SUBARU': 33,'INFINITI': 34,'LINCOLN': 35,'RENAULT': 36,'MASERATI': 37,'GMC': 38,
                        'BUICK': 39, 'DODGE': 40,'CHRYSLER': 41,'VOLVO': 42,'FIAT': 43,'სხვა': 44,'SCION': 45,'CITROEN': 46,
                        'MERCURY': 47,'ALFA ROMEO': 48,'VAZ': 49,'MOSKVICH': 50,'HAVAL': 51,'ISUZU': 52,'SATURN': 53,'DAEWOO': 54,
                        'LANCIA': 55,'DAIHATSU': 56,'GREATWALL': 57,'UAZ': 58,'SAAB': 59,'PONTIAC': 60,'SEAT': 61,'ZAZ': 62,
                        'ROVER': 63,'ROLLS-ROYCE': 64}

    color_map = {'Black': 0,'White': 1,'Silver': 2,'Grey': 3,'Blue': 4,'Orange': 5,'Brown': 6,'Carnelian red': 7,
            'Red': 8,'Green': 9,'Yellow': 10,'Beige': 11,'Golden': 12,'Pink': 13,'Sky blue': 14,'Purple': 15}
    
    df_['Manufacturer'] = df_['Manufacturer'].map(manufacturer_map)
    df_['Category'] = df_['Category'].map(category_map)
    df_['Fuel type'] = df_['Fuel type'].map(fuel_map)
    df_['Leather interior'] = df_['Leather interior'].map(leather_interior_map)
    df_['Wheel'] = df_['Wheel'].map(wheel_map)
    df_['Drive wheels'] = df_['Drive wheels'].map(drive_wheels_map)
    df_['Doors'] = df_['Doors'].map(doors_map)
    df_['Color'] = df_['Color'].map(color_map)
    df_['Gear box type'] = df_['Gear box type'].map(gear_box_map)

    def int_cols_(annot_):
        return int(annot_)

    df_['Cylinders'] = df_['Cylinders'].apply(int_cols_)
    df_['Fuel type'] = df_['Fuel type'].apply(int_cols_)
    df_['Engine volume'] = df_['Engine volume'].apply(int_cols_)
    df_['Mileage'] = df_['Mileage'].apply(int_cols_)

    model_features = pd.get_dummies(df_['Model'], drop_first = True)
    df_ = pd.concat([df_, model_features], axis = 1)
    df_ = df_.drop(['Model', 'ID'], axis = 1)

    features_path = config__['featurize']['features_path']
    df_.to_csv(features_path)

    print(f"Features have been Successfully Selected and saved to {features_path}")

    if __name__ == '__main__':

        featurize()