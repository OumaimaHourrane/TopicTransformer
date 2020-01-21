import string

input_path = 'C:/Users/Tuan-Anh Hoang/Desktop/topicRNN/cleaned_tweets.txt'
out_put_path = 'C:/Users/Tuan-Anh Hoang/Desktop/topicRNN/travel_ban'

MIN_TWEET_FREQUENCY = 10
MIN_NUMBER_KNOWN_WORDS = 10
UNK_WORD = '_unk_'

word_frequency = {}
file = open(input_path,'r')
for line in file:
    tokens = line.strip().split(' ')
    for i in range(len(tokens)):
        w = tokens[i]
        if w in word_frequency:
            word_frequency[w] += 1
        else:
            word_frequency[w] = 1
file.close()
removed_tweets = set()
while True:
    flag = True
    file = open(input_path, 'r')
    tweet_index = 0
    new_removed_tweets = []
    for line in file:
        if tweet_index in removed_tweets:
            tweet_index += 1
            continue
        tokens = line.strip().split(' ')
        num_known_words = 0
        for i in range(len(tokens)):
            w = tokens[i]
            if word_frequency[w] >= MIN_TWEET_FREQUENCY:
                num_known_words += 1
        if num_known_words < MIN_NUMBER_KNOWN_WORDS:
            removed_tweets.add(tweet_index)
            new_removed_tweets.append(tokens)
            flag = False
        tweet_index+=1
    file.close()
    if flag:
        break
    else:
        for t in range(len(new_removed_tweets)):
            tokens = new_removed_tweets[t]
            for i in range(len(tokens)):
                w = tokens[i]
                word_frequency[w] -= 1
out_file = open('%s/tweet_tokens.txt' % out_put_path, 'w')
file = open(input_path,'r')
for line in file:
    if tweet_index in removed_tweets:
        tweet_index += 1
        continue
    tokens = line.strip().split(' ')
    for i in range(len(tokens)):
        w = tokens[i]
        if word_frequency[w] >= MIN_TWEET_FREQUENCY:
            if i == 0:
                out_file.write('%s' % w)
            else:
                out_file.write(' %s' % w)
        else:
            if i == 0:
                out_file.write('%s' % UNK_WORD)
            else:
                out_file.write(' %s' % UNK_WORD)
    out_file.write('\n')
    tweet_index += 1
file.close()
out_file.close()


out_file = open('%s/tweet_vectors.txt' % out_put_path, 'w')
file = open(input_path,'r')
word_index = {}
word_index[UNK_WORD] = 0
for line in file:
    if tweet_index in removed_tweets:
        tweet_index += 1
        continue
    tokens = line.strip().split(' ')
    for i in range(len(tokens)):
        w = tokens[i]
        if word_frequency[w] >= MIN_TWEET_FREQUENCY:
            if w not in word_index:
                word_index[w] = len(word_index)
            if i == 0:
                out_file.write('%d' % word_index[w])
            else:
                out_file.write(' %d' % word_index[w])
        else:
            if i == 0:
                out_file.write('%d' % word_index[UNK_WORD])
            else:
                out_file.write(' %d' % word_index[UNK_WORD])
    out_file.write('\n')
    tweet_index += 1
file.close()
out_file.close()

out_file = open('%s/dictionary.txt' % out_put_path, 'w')
for w in word_index:
    out_file.write('%d\t%s\n' %(word_index[w], w))
out_file.close()