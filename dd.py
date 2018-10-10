import json

data_file = open('news_items.json')
data = json.load(data_file)

for news in data:
    # print news['date']
    date = news['date'].split(',')
    if 'Jan' in date[0]:
        news['date'] = date[1].strip() + '-01-' + date[0].split(' ')[1]
    elif 'Feb' in date[0]:
        news['date'] = date[1].strip() + '-02-' + date[0].split(' ')[1]
    elif 'Mar' in date[0]:
        news['date'] = date[1].strip() + '-03-' + date[0].split(' ')[1]
    elif 'Apr' in date[0]:
        news['date'] = date[1].strip() + '-04-' + date[0].split(' ')[1]
    elif 'May' in date[0]:
        news['date'] = date[1].strip() + '-05-' + date[0].split(' ')[1]
    elif 'Jun' in date[0]:
        news['date'] = date[1].strip() + '-06-' + date[0].split(' ')[1]
    elif 'Jul' in date[0]:
        news['date'] = date[1].strip() + '-07-' + date[0].split(' ')[1]
    elif 'Aug' in date[0]:
        news['date'] = date[1].strip() + '-08-' + date[0].split(' ')[1]
    elif 'Sep' in date[0]:
        news['date'] = date[1].strip() + '-09-' + date[0].split(' ')[1]
    elif 'Oct' in date[0]:
        news['date'] = date[1].strip() + '-10-' + date[0].split(' ')[1]
    elif 'Nov' in date[0]:
        news['date'] = date[1].strip() + '-11-' + date[0].split(' ')[1]
    elif 'Dec' in date[0]:
        news['date'] = date[1].strip() + '-12-' + date[0].split(' ')[1]
    print news['date']
with open("updated.json", "w") as jsonFile:
    json.dump(data, jsonFile)
    jsonFile.truncate()
    jsonFile.close()