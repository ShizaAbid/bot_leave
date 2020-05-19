from flask import Flask
app = Flask(__name__)




@app.route("/")
def home():
    return "<h1>Not Much Going On Here 5</h1>"
@app.route("/<name>")


def hello(name):
    import tensorflow as tf
    import json
    import xlrd
    import traceback   
    import numpy as np #use to handle numeric data
    import nltk #for nlp purpose
    import pandas as pd #use for file that we read
    import re #to handle regular expression
    from keras.models import load_model #To load model
    from keras import backend as K #to load the backend library that we are using 
    m = load_model('Leave_Model.h5') #loading model
    #K.clear_session()
    from tensorflow.keras.preprocessing import sequence
    from keras.preprocessing.text import Tokenizer
    from textblob import TextBlob
    from sklearn.model_selection import train_test_split
    from langdetect import detect
    import string 
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import h5py
    from datetime import datetime
    from datetime import timedelta
    import datefinder
    from dateparser.search import search_dates
    import spacy
    from textblob import TextBlob
    from nltk.tokenize import word_tokenize
    import gensim
    
    nlp = spacy.load('en_core_web_sm')

    """f = open("file.txt", "r", encoding="utf")
    text = f.read()
    f.close()
    f = open("file.txt", "w", encoding="utf")
    f.write(text)"""
    from nltk.stem.snowball import SnowballStemmer
    stemmer= SnowballStemmer("english")
    def stemming(text):
        stems =[stemmer.stem(t) for t in text]
        return stems
    def token_stems(text):
        tokens=tokenizing(text) 
        stems=stemming(tokens)
        return stems
    def tokenizing(text):
        #breaking each word and making them tokens
        tokens=[word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        #storing only alpha tokens
        filtered_tokens=[]
        for token in tokens:
            if (re.search('[a-zA-Z]|\'', token)):
                filtered_tokens.append(token)
        return filtered_tokens
    train= pd.read_excel("leave_final.xlsx")
    docs= train['Data']
    tokens = []
    for i in docs:
        temp = token_stems(i)
        tokens.append(temp)
    x, y = np.asarray(tokens) , np.asarray(train['Type'])
    xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size= 0.3, random_state=100)
    max_len=200
    max_words = 20000
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(x)



    sequences = tok.texts_to_sequences(x)
    test_sequences = tok.texts_to_sequences(xtest)
    
    raw = train['Data']
    word_tokens = [nltk.word_tokenize(str(sent)) for sent in raw]
    sent_tokens = [sent for sent in raw]
    lemmer = nltk.stem.WordNetLemmatizer()

    def LemTokens(tokens):
        return [lemmer.lemmatize(token) for token in tokens]
    
    
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    
    def LemNormalize(text):
        return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))



    def quantity_module(name):
        sub_intent = ''
        user_response = name
        user_response=user_response.lower()
        score = []
        score = quantity_model(user_response)
        if(score[0][0]>score[0][1]):
            sub_intent = sub_intent + "All"
        else:
            sub_intent = sub_intent + "Particular"
        return sub_intent


    def quantity_model(name):
        user_response = name
        train= pd.read_excel("leave_final.xlsx")
        docs= train['Data']
        tokens = []
        for i in docs:
            temp = token_stems(i)
            tokens.append(temp)  
        x, y = np.asarray(tokens) , np.asarray(train['quantity'])
        xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size= 0.1, random_state=150)
        max_len=200
        max_words = 20000
        tok = Tokenizer(num_words=max_words)
        tok.fit_on_texts(x)

        sequences = tok.texts_to_sequences(x)
        test_sequences = tok.texts_to_sequences(xtest)
            
        m = load_model('Quantity_Model.h5')
        sen = token_stems(user_response)
        sen_test = ([list(sen)])
        sen_sequences = tok.texts_to_sequences(sen_test)
        sen_sequences_matrix = sequence.pad_sequences(sen_sequences,maxlen=max_len)

                
        score = m.predict(sen_sequences_matrix)
        K.clear_session()
        #return("reach till here")
        print(user_response)
        user_response=""
        sen=""
        sen_test=""
        Intent=""
        Score=""
        print("Main_Model")
        print(score)
        return(score)



    

    def response(user_response):
        name = user_response
        robo_response=''
        sent_tokens.append(user_response)
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize,max_df = 9000,stop_words = 'english')
        tfidf = TfidfVec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        print('here')
        print(user_response)
        if(re.search('feeling sick|not feeling well',user_response)):
            sen = token_stems(user_response)
            sen_test = ([list(sen)])
            sen_sequences = tok.texts_to_sequences(sen_test)
            sen_sequences_matrix = sequence.pad_sequences(sen_sequences,maxlen=max_len)

            m = load_model('Leave_Model.h5')
            score = m.predict(sen_sequences_matrix)
            K.clear_session()
            #return("reach till here")
            print(user_response)
            user_response=""
            sen=""
            sen_test=""
            Intent=""
            Score=""
            if((score[0][0] > score [0][1])&(score[0][0] > score [0][2])&(score[0][0] > score [0][3])):
                Score = str(round(score[0][0]))
                Intent="Leave_Request"
            elif((score[0][1] > score [0][0])&(score[0][1] > score [0][2])&(score[0][1] > score [0][3])):
                Score = str(round(score[0][1]))
                Intent="Leave_Approval"
                sub_intent = leave_approval(name)
                return ('{"TopIntent": "'+Intent+'", "Percentage":'+Score + ", "+sub_intent)
            elif((score[0][2] > score [0][1])&(score[0][2] > score [0][0])&(score[0][2] > score [0][3])):
                Score = str(round(score[0][2]))
                Intent="Leave_Request"
            else:
                Score = str(round(score[0][3]))
                Intent="Leave_Inquiry"
            return ('{"TopIntent": "'+Intent+'", "Percentage":'+Score + ", ")
            
            
        elif(req_tfidf==0):
            #robo_response=robo_response+"I am sorry! I don't understand you"
            tokens=[word.lower() for sent in nltk.sent_tokenize(user_response) for word in nltk.word_tokenize(sent)]
            text = " ".join(map(str,tokens))
            if(re.search('hi|hello|how are you',text)):
                return('{"TopIntent": "Greeting"}')
            else:
                return('{"TopIntent": "None"}')
        else:
            m = load_model('Leave_Model.h5')
            sen = token_stems(user_response)
            sen_test = ([list(sen)])
            sen_sequences = tok.texts_to_sequences(sen_test)
            sen_sequences_matrix = sequence.pad_sequences(sen_sequences,maxlen=max_len)

            
            score = m.predict(sen_sequences_matrix)
            K.clear_session()
            #return("reach till here")
            print(user_response)
            user_response=""
            sen=""
            sen_test=""
            Intent=""
            Score=""
            if((score[0][0] > score [0][1])&(score[0][0] > score [0][2])&(score[0][0] > score [0][3])):
                Score = str(round(score[0][0]))
                Intent="Leave_Request"
            elif((score[0][1] > score [0][0])&(score[0][1] > score [0][2])&(score[0][1] > score [0][3])):
                Score = str(round(score[0][1]))
                Intent="Leave_Approval"
            elif((score[0][2] > score [0][1])&(score[0][2] > score [0][0])&(score[0][2] > score [0][3])):
                Score = str(round(score[0][2]))
                Intent="Leave_Request"
            else:
                Score = str(round(score[0][3]))
                Intent="Leave_Inquiry"
            return ('{"TopIntent": "'+Intent+'", "Percentage":'+Score)
            
                
           #if((round(score[0][0]*100,2))>(round(score[0][1]*100,2))):
                #Intent="Leave_Inquiry"
                #Score=str(score[0][0])
            #else:
                #Intent="Leave_Request"
                #Score=str(score[0][1])
            #return ('{"TopIntent": "'+Intent+'", "Percentage":'+Score)'''

    def name_entity_extract(name):
        name_extract = ''
        name = name.title()
        #name_func = check_name(name)
        #print(name_func)
        nlp = spacy.load('en_core_web_sm')
        name_spacy = nlp(name)
        for num,sen in enumerate(name_spacy.sents):
            for ent in sen.ents:
                if ent.label_ == 'PERSON':
                    name_func = check_name(ent.text)
                    name_extract = name_func
        if(name_extract == ''):
            name_func = check_name(name)
            name_extract = name_func
        if(name_extract != ''):
            return('"EmployeeDetail":'+'"'+str(name_extract)+'"')
        else:
            return ''


    def check_name(text):
        text = text.lower()
        print("NAME")
        print(text)
        text = ' '+text+' '
        name_database = pd.read_excel("D:\\virtual\\SoftronicEmployee.xlsx")
        emp_code = name_database['EmpCode']
        emp_id = name_database['EmpId']
        emp_name = name_database['Name']
        correct_name = ''
        variations = ''
        #print(len(name_database))
        for a in range(len(emp_name)):
            name = emp_name[a]
            msname = "ms."+name
            mrname = "mr."+name
            name = name.lower()
            name_token = [word.lower() for sent in nltk.sent_tokenize(name) for word in nltk.word_tokenize(sent)]
            if(re.search(name,text)):
                correct_name = str(emp_name[a])+"@"+str(emp_code[a])+"@"+str(emp_id[a])
            elif(re.search(msname,text)):
                correct_name = str(emp_name[a])+"@"+str(emp_code[a])+"@"+str(emp_id[a])
            elif(re.search(mrname,text)):
                correct_name = str(emp_name[a])+"@"+str(emp_code[a])+"@"+str(emp_id[a])
            else:
                for token in range(len(name_token)):
                    msname = "ms."+name_token[token]
                    mrname = "mr."+name_token[token]
                    if(re.search(' '+name_token[token]+' ',text)):
                        if(variations == ''):
                            variations = str(emp_id[a])+"@"+str(emp_code[a])+"@"+str(emp_name[a])
                        else:
                            variations = variations + ','+str(emp_id[a])+"@"+str(emp_code[a])+"@"+str(emp_name[a])
                    elif(re.search(' '+msname+' ',text)):
                        if(variations == ''):
                            variations = str(emp_id[a])+"@"+str(emp_code[a])+"@"+str(emp_name[a])
                        else:
                            variations = variations + ','+str(emp_id[a])+"@"+str(emp_code[a])+"@"+str(emp_name[a])
                    elif(re.search(' '+mrname+' ',text)):
                        if(variations == ''):
                            variations = str(emp_id[a])+"@"+str(emp_code[a])+"@"+str(emp_name[a])
                        else:
                            variations = variations + ','+str(emp_id[a])+"@"+str(emp_code[a])+"@"+str(emp_name[a])
        if(correct_name != ''):
            return(correct_name)
        else:
            return(variations)

            


            
    def date_stopwords(text):
        stopwords = nltk.corpus.stopwords.words('english')
        tokens=[word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        #print(tokens)
        filtered_sentence = []
        for w in tokens:
            if w not in stopwords:
                filtered_sentence.append(w)
        filtered_sentence = " ".join(map(str,filtered_sentence))
        return filtered_sentence


    def date_format(text):
        tokens=[word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        text = " ".join(map(str,tokens))
        corr = []
        text = nlp(text)
        text = [token.text for token in text]
        for token in range(len(text)):
            if (re.search('jan?|feb?|march|apr?|may|jun?|jul?|aug?|sep?|oct?|nov?|dec?',text[token])):
                if(re.search('january|february|march|april|may|june|july|august|september|october|november|december',text[token])):
                    print("here")
                elif(re.search('jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec',text[token])):
                    text[token]=datetime.strptime(text[token],'%b').strftime('%B')
                
                    
                if(text[token-1]!='of'):
                    #print(text[token-1])
                    if(re.search('[0-9]?',text[token-2])):
                        #print(text[token-2])
                        corr.append("of")
                corr.append(text[token].capitalize())
            else:
                corr.append(text[token])   
        string = " ".join(map(str,corr))
        return string


    def date(text):
        dates = ""
        string = date_format(text)
        print(string)
        stop_word = date_stopwords(text)
        print(stop_word)
        date = search_dates(stop_word)
        if(date == None):
            date = ""
            return date


        for match in range(len(date)):
            if((match+1) == len(date)):
                dates = dates + str(date[match])
            else:
                dates = dates + str(date[match])+"*xxx*"
        dates = dates + "++"



        text = nlp(string)
        for num,sen in enumerate(text.sents):
            for ent in sen.ents:
                is_present = False
                is_date = search_dates(ent.text)
                if ent.label_ == 'DATE':
                    dates = dates +(str(ent.text))+ "*xx*"
                    st = ent.text
                    for tok in st:
                        if(re.search('or|and|&',st)):
                            is_present = True
                    if(is_present == True):
                        dates = dates + "*uxm*" 
                    elif(len(is_date)>1):
                        dates = dates + "*uxr*"
                        
        dates = dates + "XXXXX"

        date = []


        matches = (datefinder.find_dates(string))
        for match in matches:
            date.append(match.strftime('%d-%m-%Y'))
            #dates = dates + match.strftime('%d-%m-%Y')+"*xxx*"

        string=nlp(string)


        text = [token.text for token in string]
        #print(text)
        for token in text:
            if (re.search('today|tomorrow|yesterday', token)):
                if(token == 'today'):
                    token = datetime.today().strftime('%d-%m-%Y')
                elif(token == 'yesterday'):
                    token = (datetime.now() - timedelta(days=1)).strftime('%d-%m-%Y')
                elif(token == 'tomorrow'):
                    token = (datetime.now() + timedelta(days=1)).strftime('%d-%m-%Y')
                date.append(token)
                    #dates = dates + token +"*xx*"
        date.sort(key = lambda date: datetime.strptime(date, '%d-%m-%Y'))
        for d in date:
            dates = dates + d + "*xx*"
        #print(date)
        if(dates != ""):
            return ('"datetime": "'+dates+'"')
        else:
            return ""


    def leave_approval(name):
        query = name
        sub_type = ''
        train = pd.read_excel("leave_final.xlsx")
        train = train[train['Type'] == 'Leave_Approval']
        text = train['Data']
        approve = train[train['sub_type_two'] == 'approve']
        reject = train[train['sub_type_two'] == 'reject']
       
        gen_approve = [[w.lower() for w in word_tokenize(text)] 
            for text in approve['Data']]
        gen_reject = [[w.lower() for w in word_tokenize(text)] 
                    for text in reject['Data']]

        
        dictionary_approve = gensim.corpora.Dictionary(gen_approve)
        dictionary_reject = gensim.corpora.Dictionary(gen_reject)


        
        corpus_approve = [dictionary_approve.doc2bow(gen_approve) for gen_approve in gen_approve]
        #print(corpus_leave)
        tf_idf_approve  = gensim.models.TfidfModel(corpus_approve)
        #print(tf_idf_leave)
        sims_approve = gensim.similarities.Similarity('Leave\\approve.txt',tf_idf_approve[corpus_approve],
                                              num_features=len(dictionary_approve))



        
        corpus_reject = [dictionary_reject.doc2bow(gen_reject) for gen_reject in gen_reject]
        #print(corpus_leave)
        tf_idf_reject  = gensim.models.TfidfModel(corpus_reject)
        #print(tf_idf_leave)
        sims_reject = gensim.similarities.Similarity('Leave\\reject.txt',tf_idf_reject[corpus_reject],
                                              num_features=len(dictionary_reject))


        


       
        query = name
        query_doc = [w.lower() for w in word_tokenize(query)]
        #for approval
        query_doc_bow = dictionary_approve.doc2bow(query_doc)
        query_doc_tf_idf = tf_idf_approve[query_doc_bow]
        approve = np.max(sims_approve[query_doc_tf_idf])
        print(approve)

        #for reject
        query_doc_bow = dictionary_reject.doc2bow(query_doc)
        query_doc_tf_idf = tf_idf_reject[query_doc_bow]
        reject = np.max(sims_reject[query_doc_tf_idf])
        print(reject)

       
        if(approve > reject):
            sub_type = sub_type + "approve"
        else:
            sub_type = sub_type + "reject"

        if(re.search('not approve|not accept|not acept|reject|disapprove|dis approve',name)):
            sub_type = "disapprove"
        elif(re.search('approve|accept|acept',name)):
            sub_type = "approve"
        sub_type = '"Leave_Approval":' + '"' + sub_type + '"'
        return(sub_type)




    def leavetype(text):
        tokens=[word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        #storing only alpha tokens
        filtered_tokens=[]
        for token in range(len(tokens)):
            if (re.search('earned|privilege|casual|sick|medical|half-pay|casual|maternity|quarantine|study|sabbatical|halfday|annual', tokens[token])):
                #next = token + 1
                #if(next < len(tokens)):
                    #if(tokens[next] == 'leave'):
                filtered_tokens.append(str(tokens[token]+' leave').title()) #filtered_tokens #+ str(tokens[token]+' leave').title()+"*xx*"
                        #token +=1
        leavetypes=""
        if(len(filtered_tokens) != 0):
            if(len(filtered_tokens)>1):
                for leaves in range(len(filtered_tokens)-1):
                    leavetypes = leavetypes + filtered_tokens[leaves] + "*xx*"
                #leavetypes = leavetypes + filtered_tokens[len(filtered_tokens)]
            leavetypes = leavetypes + filtered_tokens[(len(filtered_tokens))-1]
            return ('"Leave_Type": "'+leavetypes+'"')
        else:
            return ""

    def for_model(user_response,date,leavetype,name_entity,approval):
        user_response=user_response.lower()
        res = response(user_response)
        if(re.search('{"TopIntent": "None"}|{"TopIntent": "Greeting"}',res)):
            f.close()
            K.clear_session()
            return (res+date+leavetype+name_entity+self)
        else:
            res=res+',"Entities": {'
            print ("**********************************************")
            print ("**********************************************")
            print (res)
            ans = ''
            ans = ans+res
            flag = 0
            print (ans)            
            if(date != ""):
                flag = 1
                print(date)
                ans = ans +date
            if(leavetype != ""):
                if (flag == 1):
                    ans = ans +','
                flag = 2
                ans = ans +leavetype
            if(name_entity != ""):
                if ((flag == 1)|(flag == 2)):
                    ans = ans + ','
                flag = 3
                ans = ans + name_entity
            if(re.search("Leave_Approval",res)):
                if ((flag == 1)|(flag == 2)|(flag == 3)|(flag == 4)):
                    ans = ans + ','
                ans = ans + approval
                
            ans = ans + '}}'
            #f.write(name + "\n")
            #f.close()
            K.clear_session()
            return(ans)


    self = '"Self":'
    quant =  quantity_module(name)
    approval = leave_approval(name)
    leavetype = leavetype(name)
    date = date(name)
    text = nlp(name)
    name_entity = name_entity_extract(name)
        
    text = [token.text for token in text]
    print('length text')
    print(len(text))
    if(len(text)< 3):
        dec = TextBlob(name)
        if((dec.detect_language())== 'en'):
            K.clear_session()
            return('{"TopIntent": "None"'+',"Entities": {'+date+leavetype+name_entity+'}}')
        else:
            K.clear_session()
            return('{"TopIntent": "NoneLang"'+',"Entities": {'+date+leavetype+name_entity+'}}')
    else:
        user_response = name
        dec = TextBlob(user_response)
        if((dec.detect_language()) == 'en'):
            date_format = date_format(name)
            #print(date)
            text = nlp(date_format)
            found = True
            temp = ""
            for num,sen in enumerate(text.sents):
                #print(sen.text)
                for ent in sen.ents:
                    #print(ent.text)
                    temp = temp + " "+ent.text
                    is_date = search_dates(ent.text)
                    if(is_date == None):
                        found = False
                        is_date = '0'
            if(found == True):
                if(len(temp)+2 > len(name)):
                    K.clear_session()
                    return('{"TopIntent": "None"'+',"Entities": {'+date+leavetype+name_entity+'}}')
                else:
                    return(for_model(name,date,leavetype,name_entity,approval))
               
            else:
                 return(for_model(name,date,leavetype,name_entity,approval))

        else:
            f.close()
            K.clear_session()
            return ('{"TopIntent": "NoneLang"'+',"Entities": {'+date+leavetype+name_entity+'}}')
            
    #return "<h1> Here 5</h1>"+name
    K.clear_session()
app.run(host='127.0.0.1', port=60948)

