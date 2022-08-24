import pandas as pd
import numpy as np
import serial
import binascii

csv_file = 'output/1_227_2047.txt'

com_number='COM23'
com_baud_rate=115200

if __name__ == '__main__':
    com_session = serial.Serial(com_number, com_baud_rate)
    print('Opened: '+com_session.portstr)
    datas = pd.read_csv(csv_file)
    datas = np.array(datas.values,'uint8') #转换为uint8类型

    for i in datas:
        for j in i:
            com_session.write(binascii.a2b_hex(f'{j:02X}'))

    com_session.close()

