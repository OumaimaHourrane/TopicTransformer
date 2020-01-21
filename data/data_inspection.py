import operator

data_path = 'C:/Users/Tuan-Anh Hoang/Desktop/tss/dataset/travel_ban/travel_ban.txt'
output_path = 'C:/Users/Tuan-Anh Hoang/Desktop/topicRNN'

def count_raw_word():
    word_count = {}
    with open(data_path, 'r', encoding="utf8") as f:
        for line in f:
            tokens = line.lower().split('\t')[4].strip().split(' ')
            for word in tokens:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1
    out_file = open('%s/raw_word_count.csv' % output_path,'w')
    word_count = sorted(word_count.items(), key=operator.itemgetter(1))
    for word in word_count:
        out_file.write('\"%s\",%d\n' %(word[0].encode('utf-8').strip(),word[1]))
    out_file.close()

def count_preprocessed_word():
    word_count = {}
    with open('%s/cleaned_tweets.txt' % output_path,'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            for word in tokens:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1
    out_file = open('%s/preprocessed_word_count.csv' % output_path,'w')
    word_count = sorted(word_count.items(), key=operator.itemgetter(1))
    for word in word_count:
        out_file.write('%s\t%d\n' %(word[0], word[1]))
    out_file.close()

#count_raw_word()
count_preprocessed_word()