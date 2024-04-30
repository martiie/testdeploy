import time
from datetime import datetime
import os
import csv
import pandas as pd

class Record:
    def __init__(self) -> None:
          pass

    def record_to_csv(self,face_crops):
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("attend/Attend_"+date+".csv")
        try:
            if exist:
                rr = pd.read_csv("attend/Attend_"+date+".csv")
                if str(face_crops[0]['name']).split('.')[0] not in list(rr['NAME']) and str(face_crops[0]['name']).split('.')[0] != 'Unknown':
                    with open("attend/Attend_"+str(date)+".csv",'+a',newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([timestamp,str(face_crops[0]['name']).split('.')[0]])
            else:
                with open("attend/Attend_"+str(date)+".csv",'w',newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['TIME','NAME'])
                    writer.writerow([timestamp,str(face_crops[0]['name']).split('.')[0]])
        except:
            pass

