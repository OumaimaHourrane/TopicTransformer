import string
import re

data_path = 'C:/Users/Tuan-Anh Hoang/Desktop/tss/dataset/travel_ban/travel_ban.txt'
output_path = 'C:/Users/Tuan-Anh Hoang/Desktop/murnn/cleaned_tweets.txt'

#NAME_EXP = r'(^[a-zA-Z]\.\s)|(\s[a-zA-Z]\.\s)|(\s[a-zA-Z]\.$)|(^[a-zA-Z]\.$)'
NAME_EXP = r'\s[a-zA-Z]\.\s'
MONEY_EXP = r'(([\d][\d,]+\.?\d*[kKmMbB]*)*)[\$€£]{1}([\d][\d,]+\.?\d*[kKmMbB]*)'
PERCENTAGE_EXP = r'(?<!\.)(?!0+(?:\.0+)?%)(?:\d|[1-9]\d|100)(?:(?<!100)\.\d+)?%'
URL_EXP = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
EMAIL_EXP = r'\w[\w\._]+@[\w]+[\w\.]+'
RETWEET_EXP = r'rt @\S+:'
NUMBER_EXP = r'([-\s]+[\d][\d,]*\.?\d*[kKmMbB]*[\s])|(^[-]?[\d][\d,]*\.?\d*[kKmMbB]*[\s])|(-?[\d][\d,]*\.?\d*[kKmMbB]*$)'
PRINTABLE_CHARACTERS = set(string.printable+'$€£')
INVALID_FIRST_CHARACTERS = {'\'', '`', '~', '!', '^', '&', '*', '-', '+', ':', ';', '?', ',', '.', '/', '\\'}

def filter_unicode_char(tweet):
    o = filter(lambda x: x in PRINTABLE_CHARACTERS, tweet)
    return ''.join(o)

def remove_html_code(tweet):
    tweet = tweet.replace('&amp;quot;', ' " ')
    tweet = tweet.replace('&amp;apos;', ' \' ')
    tweet = tweet.replace('&amp;gt;', ' > ')
    tweet = tweet.replace('&amp;lt;', ' < ')
    tweet = tweet.replace('&amp;lsquo;', ' \' ')
    tweet = tweet.replace('&amp;rsquo;s', ' \'s ')
    tweet = tweet.replace('&amp;rsquo;', ' \' ')
    tweet = tweet.replace('&amp;', ' & ')
    tweet = tweet.replace('&quot;', ' " ')
    tweet = tweet.replace('&apos;', ' \' ')
    tweet = tweet.replace('&gt;', ' > ')
    tweet = tweet.replace('&lt;', ' < ')
    tweet = tweet.replace('&lsquo;', ' \' ')
    tweet = tweet.replace('&rsquo;s', ' \'s ')
    tweet = tweet.replace('&rsquo;', ' \' ')
    tweet = tweet.replace('&gt;', ' > ')
    tweet = tweet.replace('&lt;', ' < ')
    tweet = tweet.replace('&039;', '\'')
    return tweet


def clean_tweet(tweet):
    tweet = tweet.lower()
    tweet = tweet.replace('…', '')
    tweet = remove_html_code(tweet)
    tweet = tweet.replace(' .@', ' @')
    tweet = re.sub(RETWEET_EXP, ' ', tweet)
    tweet = re.sub(URL_EXP, ' _url_symbol_ ', tweet)
    tweet = re.sub(EMAIL_EXP, ' _email_symbol_ ', tweet)

    tweet = tweet.replace('\"', ' \" ')
    tweet = re.sub('~+', ' ', tweet)
    tweet = re.sub('\^+', ' ', tweet)
    tweet = re.sub('\.+', '.', tweet)
    tweet = re.sub(',+', ',', tweet)
    tweet = re.sub(':+', ' : ', tweet)
    tweet = re.sub('@+', '@', tweet)
    tweet = re.sub('#+', '#', tweet)
    tweet = re.sub('%+', '%', tweet)
    tweet = re.sub('\*+', '*', tweet)
    tweet = re.sub('-+', '-', tweet)
    tweet = re.sub('_+', '_', tweet)
    tweet = re.sub('\++', '+', tweet)
    tweet = re.sub('\?+', ' ? ', tweet)
    tweet = re.sub('!+', ' ! ', tweet)
    tweet = re.sub('\(+', ' ( ', tweet)
    tweet = re.sub('\)+', ' ) ', tweet)
    tweet = re.sub('<+', ' < ', tweet)
    tweet = re.sub('>+', ' > ', tweet)
    tweet = re.sub('{+', ' { ', tweet)
    tweet = re.sub('}+', ' } ', tweet)
    tweet = re.sub('\[+', ' [ ', tweet)
    tweet = re.sub('\]+', ' ] ', tweet)
    tweet = re.sub('#+', '#', tweet)

    tweet = tweet.replace(' *', ' ')
    tweet = tweet.replace('* ', ' ')
    tweet = re.sub(r' \\\'', ' \' ', tweet)
    tweet = re.sub(r'\\\' ', ' \' ', tweet)
    tweet = tweet.replace('\'s ', ' _possessive_symbol_ ')
    tweet = re.sub(r'[\W\s]+\'', ' \' ', tweet)
    tweet = re.sub(r'\'[\W\s]+', ' \' ' , tweet)
    #tweet = tweet.replace('\' ', ' ')
    tweet = tweet.replace(' -', ' - ')
    tweet = tweet.replace('- ', ' - ')
    if(tweet[0] in INVALID_FIRST_CHARACTERS):
        tweet = tweet[1:-1]

    tweet = re.sub(MONEY_EXP, ' _money_amount_symbol_ ', tweet)
    tweet = re.sub(PERCENTAGE_EXP, ' _percentage_symbol_ ', tweet)
    tweet = re.sub(NUMBER_EXP, ' _number_symbol_ ', tweet)
    all_name_matches = re.findall(NAME_EXP, tweet)
    print(all_name_matches)
    for m in all_name_matches:
        tweet = tweet.replace(m, '%s ' % m[:-2])

    tweet = re.sub('\.+', ' . ', tweet)
    tweet = re.sub(',+', ' , ', tweet)

    tweet = re.sub('\s+', ' ', tweet)

    tweet = tweet.replace( ' non-', ' non ')
    tweet = re.sub('^non-', ' non ', tweet)

    tweet = re.sub('\W+u\s*\.?\s*s\s*\W+', ' us ', tweet)
    tweet = re.sub('^u . s . ', ' us ', tweet)

    tweet = tweet.replace(' u . s ', ' us ')
    tweet = re.sub('^u . s ', ' us ', tweet)

    tweet = re.sub('\W+u\s*\.?\s*k\s*\W+', ' uk ', tweet)
    tweet = tweet.replace(' u . k . ', ' uk ')
    tweet = re.sub('^u . k . ', ' uk ', tweet)

    tweet = tweet.replace(' u . k ', ' uk ')
    tweet = re.sub('^u . k ', ' uk ', tweet)

    tweet = tweet.replace('a . m ', 'am ')
    tweet = tweet.replace('p . m ', 'pm ')

    tweet = re.sub('\W+e\s*\.?\s*g\s*\W+', ' eg ', tweet)

    tweet = filter_unicode_char(tweet)

    tweet = re.sub('(\. )+', '. ', tweet)
    tweet = re.sub('(, )+', ', ', tweet)
    tweet = re.sub('(: )+', ' : ', tweet)
    tweet = re.sub(' # ', ' ', tweet)
    tweet = re.sub('(% )+', '% ', tweet)
    tweet = re.sub('(\* )+', '* ', tweet)

    tweet = re.sub('(- )+', ' ', tweet)
    tweet = re.sub('_ ', ' ', tweet)
    tweet = re.sub(' -', ' ', tweet)
    tweet = re.sub('(\+ )+', '+ ', tweet)
    tweet = re.sub('(\? )+', '? ', tweet)
    tweet = re.sub('(! )+', '! ', tweet)
    tweet = re.sub('(\( )+', '( ', tweet)
    tweet = re.sub('(\) )+', ') ', tweet)
    tweet = re.sub('(< )+', '< ', tweet)
    tweet = re.sub('(> )+', '> ', tweet)
    tweet = re.sub('({ )+', '{ ', tweet)
    tweet = re.sub('(} )+', '} ', tweet)
    tweet = re.sub('(\[ )+', '[ ', tweet)
    tweet = re.sub('(\] )+', '] ', tweet)
    tweet = re.sub(';+', ' ; ', tweet)
    tweet = tweet.replace('(# )+', '# ')

    tweet = re.sub('\s+', ' ', tweet)
    tweet = tweet.strip()
    return tweet

out_file = open(output_path,'w')
with open(data_path, 'r', encoding="utf8") as f:
    for line in f:
        print(line)
        tweet = line.split('\t')[4]
        #out_file.write('%s\n' % tweet.encode('utf-8').strip())
        tweet = clean_tweet(tweet)
        out_file.write('%s\n' %tweet)
out_file.close()
print('done')

# s = 'RT @nytopinion: Donald J. Trump is a pathological liar. Say it. Write it. Never become inured to it, writes @CharlesMBlow. https://t.co/Il0…'
# print(s)
# print(clean_tweet(s))