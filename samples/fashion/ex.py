import os
import sys
import json
import glob
from collections import ChainMap

json_list = []

DATASET_DIR = '/Users/Yeonhwa/Desktop/K-Fashion 이미지'
dataset_dir = os.path.join(DATASET_DIR, "Training")
dataset_dir = os.path.join(dataset_dir,"라벨링데이터")

item=["아우터","하의","원피스","상의"]

codi = ["클래식","프레피",
        "매니시","톰보이",
        "페미닌","로맨틱","섹시",
        "히피","웨스턴","오리엔탈",
        "모던","소피스트케이티드","아방가르드",
        "컨트리","리조트",
        "젠더리스",
        "스포티",
        "레트로","키치","힙합","펑크",
        "밀리터리","스트리트"]

for j in codi:
    json_dir = os.path.join(dataset_dir,j)
    labeling_data_list=glob.glob(json_dir+'/*.json')
    for i in labeling_data_list:
        file_path = i

        with open(file_path, 'r', encoding='UTF-8') as f:
            json_data = json.load(f)
            image_id = json_data["이미지 정보"]["이미지 파일명"]
            print(image_id)
            polygons=[]
            for m in item:
                q=json_data["데이터셋 정보"]["데이터셋 상세설명"]["라벨링"][m]
                value=q[0].get("카테고리")
                if value==None:
                    continue
                else:
                    polygons.append(q[0]['카테고리'])
                    print("i'm present")
            print(polygons)
