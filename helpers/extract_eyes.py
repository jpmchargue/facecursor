# Extract only eye landmarks from the IBUG XML files.
import sys

index_low = 36
index_high = 48

if len(sys.argv) == 2:
    xmlfile = sys.argv[1]
    oldlines = []
    with open(xmlfile) as file:
        oldlines = file.readlines()

    newlines = []
    parts_seen = 0
    for line in oldlines:
        if "<part" in line:
            if parts_seen >= index_low and parts_seen < index_high:
                newlines.append(line)
            parts_seen += 1
        else:
            newlines.append(line)
            parts_seen = 0

    with open("output.xml", 'w') as output:
        output.writelines(newlines)
