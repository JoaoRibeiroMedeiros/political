
# %%

import pandas as pd

deputados = pd.read_csv('Deputados2018.csv')
deputados['twitter'] = range(513)

from time import sleep
from selenium import webdriver
from selenium.webdriver.common.keys import Keys



def ScrapaDeputados2018(deputados, username, senha):
    
    deputadas = []
    for i in deputados.Nome:
        if i.split()[0].strip('.')[-1]=='a' and i.split()[0] not in ['Zeca', 'Tiririca', 'Gonzaga', 'Átila', 'Guiga']:
            deputadas.append(i)

    browser = webdriver.Chrome('/Users/joao/Dropbox/crawler/chromedriver')
    browser.get('https://www.twitter.com/login')

    sleep(5)

    #insere senha

    xpath='//*[@id="react-root"]/div/div/div[2]/main/div/div/div[2]/form/div/div[2]/label/div/div[2]/div/input'
    
    browser.find_element_by_xpath(xpath).send_keys(senha)  

    #username

    xpath='//*[@id="react-root"]/div/div/div[2]/main/div/div/div[2]/form/div/div[1]/label/div/div[2]/div/input'
    
    browser.find_element_by_xpath(xpath).send_keys(username)
    
    #entra

    browser.find_element_by_xpath('//*[@id="react-root"]/div/div/div[2]/main/div/div/div[2]/form/div/div[3]/div/div').click()

    sleep(3)

    nomes = []
    handles = []

    for i in deputados.Nome:

        #clica em busca

        xpath='//*[@id="react-root"]/div/div/div[2]/header/div/div/div/div[1]/div[2]/nav/a[2]/div'
        browser.find_element_by_xpath(xpath).click()

        sleep(2)

        deputado = i

        if i in deputadas:
            complementobusca = ' deputada'
        else:
            complementobusca = ' deputado'

        #preenche nome do deputado ou deputada

        xpath='//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/div[1]/div[1]/div/div/div/div/div[1]/div[2]/div/div/div/form/div[1]/div/div/div[2]/input'
        browser.find_element_by_xpath(xpath).send_keys(deputado + complementobusca)
        browser.find_element_by_xpath(xpath).send_keys(Keys.ENTER)

        sleep(2)

        #clica na aba people
        xpath='//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/div[1]/div[2]/nav/div/div[2]/div/div[3]/a/div/span'
        browser.find_element_by_xpath(xpath).click()

        sleep(4)
        xpath='//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/div[2]/div/div/div/div[1]/span'
        try: #se nao achar resultado
            if browser.find_element_by_xpath(xpath).text == 'No results for "' +deputado + complementobusca+ '"':
                continue
        except:   
            xpath = '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/div[2]/div/div/section/div/div/div[1]/div/div'
            browser.find_element_by_xpath(xpath).click()
            
            #clica no resultado da pesquisa
            
            nomes.append(i) 
            
            #salva o url, onde tem o handle
            
            handle = browser.current_url
            print(handle)
            handles.append(handle)
            
            #guarda no df deputados
            
            deputados.loc[deputados[deputados.Nome == i].index[0],'twitter'] = handle[20:]

    browser.close()
    
    return deputados, handles, nomes

