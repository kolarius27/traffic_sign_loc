#!/usr/bin/env python
# -*- coding: utf-8 -*
import json
import os


test_data = r'C:\Users\micha\Desktop\Gekon\znacky\ukázka dat\trenovaci_data.txt'
ximilar_file = r'C:\Users\micha\Desktop\Gekon\znacky\ukázka dat\1310899080_550000_359_1_01.txt'
ximilar_folder = r'C:\Users\micha\Desktop\Gekon\znacky\ukázka dat\ximilar_detect'

def delete_comments(ximilar_file):
    ximilar_fixed = ximilar_file.replace('.txt', '_f.txt')
    with open(ximilar_file, 'r') as file:
        split_line = [line.split("//")[0].strip() for line in file.readlines()]
        print(split_line)
        json_str = "\n".join(split_line)
        json_data = json.loads(json_str)
        print(json_data)
    with open(ximilar_fixed, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


def create_json(x, y, list_sign):
    print(type(list_sign))
    list_sign_json = []
    for sign in list_sign:
        list_sign_json.append({
                        "name": "traffic sign",
                        "bound_box": [
                            135,
                            248,
                            163,
                            336
                        ],
                        "prob": 0.9431861639022827,
                        "traffic sign code": sign
                    })
        
    json_data = {
                "name": "nosic",
                "bound_box": [
                    1390,
                    348,
                    1414,
                    1436
                ],
                "Type": [
                    {
                        "name": "vlastni konstrukce",
                        "prob": 0.9823
                    }
                ],
                "prob": 0.9562318616,
                "points": [
                    {
                        "name": "pole bottom",
                        "point": [
                            int(x),
                            int(y)
                        ],
                        "prob": 0.9431861639022827
                    },
                    {
                        "name": "pole top",
                        "point": [
                            105,
                            348
                        ],
                        "prob": 0.9431861639022827
                    }
                ],
                "traffic signs": list_sign_json
            }
    
    return json_data




def create_ximilar(test_data, ximilar_folder):
    print('here')
    with open(test_data, 'r') as f:
        print('here')
        i=0
        f_readlines = f.readlines().copy()
        reference_name = f_readlines[0].split(', ')[0]
        print(reference_name)
        for line in f_readlines:
            print(line)
            split_line = line.split(', ')
            if split_line[0] == reference_name:
                i+=1
            else:
                i=1
            json_data = create_json(split_line[1], split_line[2], split_line[3].strip('\n').strip('][').split(', '))
            json_path = os.path.join(ximilar_folder, split_line[0].replace('.jpg', '_{}.txt'.format(i)))

            print(json_path)
            with open(json_path, 'w', encoding='utf-8') as g:
                json.dump(json_data, g, ensure_ascii=False, indent=4)
            reference_name = split_line[0]



if __name__ == '__main__':
    #delete_comments(ximilar_file)
    create_ximilar(test_data, ximilar_folder)