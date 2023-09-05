import joblib
import pandas as pd
import re
import string

DT = joblib.load('BestClf.joblib')
vectorization = joblib.load('vectorization.joblib')


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_DT = DT.predict(new_xv_test)

    return pred_DT


if __name__ == '__main__':
    news = '''
    UNITED NATIONS (Reuters) - Two North Korean shipments to a Syrian government agency responsible for the country s chemical weapons program were intercepted in the past six months, according to a confidential United Nations report on North Korea sanctions violations. The report by a panel of independent U.N. experts, which was submitted to the U.N. Security Council earlier this month and seen by Reuters on Monday, gave no details on when or where the interdictions occurred or what the shipments contained.   The panel is investigating reported prohibited chemical, ballistic missile and conventional arms cooperation between Syria and the DPRK (North Korea),  the experts wrote in the 37-page report.   Two member states interdicted shipments destined for Syria. Another Member state informed the panel that it had reasons to believe that the goods were part of a KOMID contract with Syria,  according to the report. KOMID is the Korea Mining Development Trading Corporation. It was blacklisted by the Security Council in 2009 and described as Pyongyang s key arms dealer and exporter of equipment related to ballistic missiles and conventional weapons. In March 2016 the council also blacklisted two KOMID representatives in Syria.   The consignees were Syrian entities designated by the European Union and the United States as front companies for Syria s Scientific Studies and Research Centre (SSRC), a Syrian entity identified by the Panel as cooperating with KOMID in previous prohibited item transfers,  the U.N. experts wrote.  SSRC has overseen the country s chemical weapons program since the 1970s. The U.N. experts said activities between Syria and North Korea they were investigating included cooperation on Syrian Scud missile programs and maintenance and repair of Syrian surface-to-air missiles air defense systems. The North Korean and Syrian missions to the United Nations did not immediately respond to a request for comment.  The experts said they were also investigating the use of the VX nerve agent in Malaysia to kill the estranged half-brother of North Korea s leader Kim Jong Un in February.  North Korea has been under U.N. sanctions since 2006 over its ballistic missile and nuclear programs and the Security Council has ratcheted up the measures in response to five nuclear weapons tests and four long-range missile launches. Syria agreed to destroy its chemical weapons in 2013 under a deal brokered by Russia and the United States. However, diplomats and weapons inspectors suspect Syria may have secretly maintained or developed a new chemical weapons capability. During the country s more than six-year long civil war the Organisation for the Prohibition of Chemical Weapons has said the banned nerve agent sarin has been used at least twice, while the use of chlorine as a weapon has been widespread. The Syrian government has repeatedly denied using chemical weapons. 
    '''

    print(manual_testing(news))
