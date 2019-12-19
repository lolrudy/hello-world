import csv

pred1 = []
with open("./data/submit.csv") as csvfile:
    csv_reader = csv.reader(csvfile)  # Ê¹ÓÃcsv.reader¶ÁÈ¡csvfileÖÐµÄÎÄ¼þ
    header = next(csv_reader)  # ¶ÁÈ¡µÚÒ»ÐÐÃ¿Ò»ÁÐµÄ±êÌâ
    for row in csv_reader:  # ½«csv ÎÄ¼þÖÐµÄÊý¾Ý±£´æµ½birth_dataÖÐ
        pred1.append(int(row[1]))

pred2 = []
with open("./data/submit1.csv") as csvfile:
    csv_reader = csv.reader(csvfile)  # Ê¹ÓÃcsv.reader¶ÁÈ¡csvfileÖÐµÄÎÄ¼þ
    header = next(csv_reader)  # ¶ÁÈ¡µÚÒ»ÐÐÃ¿Ò»ÁÐµÄ±êÌâ
    for row in csv_reader:  # ½«csv ÎÄ¼þÖÐµÄÊý¾Ý±£´æµ½birth_dataÖÐ
        pred2.append(int(row[1]))

with open("./data/evaulate.csv", 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)  # Ê¹ÓÃcsv.reader¶ÁÈ¡csvfileÖÐµÄÎÄ¼þ
    csv_writer.writerow(['image_id', 'label1', 'label2'])
    for index in range(5000):
	    if pred1[index] != pred2[index]:
	    	csv_writer.writerow([index, pred1[index], pred2[index]])
