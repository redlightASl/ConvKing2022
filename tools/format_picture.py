import cv2
import pandas as pd

SHOW_PICTURE_STATE = False
PRINT_DATA = False

file_name = '1_227_2047'

if __name__ == '__main__':
    with open('output/'+file_name+'.txt', 'w') as fp:
        img = cv2.imread('source_picture/'+file_name+'.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if SHOW_PICTURE_STATE:
            for i in img:
                for j in i:
                    print(f'{j}', end='\t', file=fp)
                print('\n', end='', file=fp)
        elif(PRINT_DATA):
            for i in img:
                for j in i:
                    print(f'{j}', end='', file=fp)
        else:
            csv_data = pd.DataFrame(img)
            csv_data.to_csv('output/'+file_name+'.txt', index=False)

        print('Done')
