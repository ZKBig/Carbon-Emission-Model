# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-05-29-4:20 下午

import pandas as pd
import numpy as np


def data_processing(numpy_data, step=5):
    data = [numpy_data[i:i + step] for i in range(0, len(numpy_data), step)]
    list = []

    for x in data:
        a = np.sum(x, axis=0)
        a = a / 5
        list.append(a)

    return list


Car_emission = pd.read_excel(r'CO2_emission_data.xlsx', sheet_name='Sheet2')
car_v = np.array(Car_emission.iloc[:, 1])
car_a = np.array(Car_emission.iloc[:, 2])
car_vsp = np.array(Car_emission.iloc[:, 3])
car_CO2 = np.array(Car_emission.iloc[:, 4])

v_list = []
a_list = []
vsp_list = []
CO2_list = []

v_list = data_processing(car_v)
a_list = data_processing(car_a)
vsp_list = data_processing(car_vsp)
CO2_list = data_processing(car_CO2)

pre_excel = {'v': v_list, 'a': a_list, 'vsp': vsp_list, 'CO2': CO2_list}

pf = pd.DataFrame(pre_excel)

pf.to_excel('./苏州金龙.xlsx', encoding='utf-8')


