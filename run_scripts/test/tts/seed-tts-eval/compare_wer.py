import sys
import json
import csv


csv_file1 = sys.argv[1]
csv_file2 = sys.argv[2]

with open(csv_file1, "r") as f1, open(csv_file2, "r") as f2:
    reader1 = list(csv.reader(f1, delimiter='\t'))
    reader2 = list(csv.reader(f2, delimiter='\t'))
    for line1, line2 in zip(reader1, reader2):
        #import pdb; pdb.set_trace()
        if len(line1) <= 1 or len(line2) <= 1:
            #continue
            import pdb; pdb.set_trace()
        wer1 = float(line1[1])
        wer2 = float(line2[1])
        if wer1 < wer2 and wer1 < 0.000000001:
            print(f"wav={line1[0]}, ref_text={line1[2]}, response1={line1[3]}, wer1={line1[1]}, response2={line2[3]}, wer2={line2[1]}")
