# -*- coding: utf-8 -*-
# @Time : 2021/8/30 9:48
# @FileName: GetLOFData.py

__author__ = 'Zhigen.li'

import os
import time
import logging
from selenium import webdriver
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import font_manager
from datetime import datetime
from selenium.common import exceptions as ex

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class Stock(object):
    def __init__(self, con):
        # 无浏览器界面化
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        self.brower = webdriver.Chrome(options=options)
        self.con = con

    def getAllStockUrl(self):
        if os.path.exists(self.con.url_path):
            return

        url = 'https://fund.eastmoney.com/'
        self.brower.get(url)

        data = self.brower.find_element_by_xpath('//*[@id="jjjz"]/div[4]/table/tfoot/tr/td/a')

        data_information = data.get_attribute('href')
        time.sleep(2)
        self.brower.get(data_information)

        table_data = {}
        fp = open(self.con.url_path, 'w')

        # find_element寻找第一个 find_elements寻找所有的
        for i in range(int(self.brower.find_element_by_xpath('//*[@id="pager"]/span[9]').text[1:-1])):
            try:
                tags = self.brower.find_elements_by_xpath('//*[@id="oTable"]/tbody/tr')
                for i in tags:
                    name = i.find_element_by_xpath('./td[5]/nobr/a[1]').text
                    num = i.find_element_by_xpath('./td[5]/nobr/a[1]').get_attribute('href')
                    num = num[:-5]
                    if name not in list(table_data.keys()):
                        fp.write(name + ',' + 'http://fundf10.eastmoney.com/jjjz_{}.html'.format(num[-6:]) + '\n')
                        table_data.setdefault(name, 'http://fundf10.eastmoney.com/jjjz_{}.html'.format(num[-6:]))
                        logging.info('{} --> {}'.format(name, table_data[name]))

                self.brower.find_element_by_xpath('//*[@id="pager"]/span[8]').click()
                fp.flush()
                time.sleep(3)
            except ex.StaleElementReferenceException as e:
                self.brower.find_element_by_xpath('//*[@id="pager"]/span[8]').click()
                time.sleep(4)
                logging.warning(str(e))
        self.brower.close()
        fp.close()

    def _getdata(self, url):
        self.brower.get(url)
        next_page = 1
        # 获取文件名
        table_name = self.brower.find_element_by_xpath('//*[@id="jzpng"]').get_attribute('alt')[:-4]
        # 获取总页数
        all_page = int(self.brower.find_element_by_xpath('//*[@id="pagebar"]/div[1]/label[7]').text)
        tables = self.brower.find_element_by_xpath('//div[@class="txt_in"]/div[2]/div/div[2]')
        line_menu = [i for j in tables.text.split('\n') for i in j.split(" ")][:6]

        # 数据处理
        df = pd.DataFrame()
        for i in line_menu:
            df[i] = pd.Series(dtype='float64')

        while (next_page <= all_page):
            tables = self.brower.find_elements_by_xpath('//div[@class="txt_in"]/div[2]/div/div[2]//tbody/tr')
            for table in tables:
                line_data = []
                for i in table.find_elements_by_xpath('./td')[:6]:
                    line_data.append(i.text)
                if len(line_data) == len(line_menu):
                    df = df.append(pd.Series(line_data, index=line_menu), ignore_index=True)

            next_page += 1
            if next_page <= all_page:
                self.brower.find_element_by_xpath(
                    '//*[@id="pagebar"]/div[1]/label[@value="{}"][2]'.format(next_page)).click()
            time.sleep(5)

            if all_page > 10:
                if next_page % 10 == 0:
                    logging.info("\033[31m正在获取 \033[36;1m{}\033[31m 数据，已完成：{} {}% \033[0m".format(table_name, '▊' * int(
                        next_page / all_page * 10), round(next_page / all_page * 100, 2)))

        logging.info("\033[31m正在获取 \033[36;1m{}\033[31m 数据：已完成，{} {}% \033[0m".format(table_name, '▊' * 10, 100))
        self.brower.close()
        days = []
        sume = 0
        lists = []

        for i in range(len(df['日增长率'])):
            temp = df['日增长率'][i][:-1]  # 去掉%
            value = round(float(temp), 2) * 0.01 if temp != '-' else 0
            if value > 0:
                j = 1
                sume += 1
            elif value == 0:
                j = 0
                sume = sume
            else:
                j = -1
                sume -= 1

            days.append(j)
            lists.append(sume)
            df['日增长率'][i] = value

        df["持续天数"] = pd.Series(days)
        dataPath = self.con.data_path + table_name[:-8] + '_{}.csv'.format(datetime.now().strftime('%Y%m%d'))
        df.to_csv(dataPath, encoding='gbk', index=False)

        if len(df['日增长率']) < 100:
            return df, lists, len(df['日增长率']), df['净值日期'], table_name
        return df, lists[:100], 100, df['净值日期'][:100], table_name

    def drawpict(self, name):
        with open(self.con.url_path, 'r') as fp:
            csvLines = fp.read().split('\n')

        url = None
        stockName = None
        for csvLine in csvLines:
            if name in csvLine:
                csvCol = csvLine.split(',')
                stockName = csvCol[0]
                url = csvCol[1]
                break
        if url == None:
            raise Exception("网站暂未收录 {}".format(name))
        logging.info(url)
        dataPath = self.con.data_path + stockName + '_{}.csv'.format(datetime.now().strftime('%Y%m%d'))
        if os.path.exists(dataPath):
            return stockName

        df, lists, numn, data, table_name = self._getdata(url)
        my_font = font_manager.FontProperties(fname="./fonts/simsun.ttc", size=15)
        plt.figure(figsize=(18, 9))
        # 处理连续增长天数
        plt.plot(range(numn), df['日增长率'][numn - 1::-1] * 100, label=u"日增长率")

        d_start, d_end = str(df['净值日期'][numn - 1])[:10], str(df['净值日期'][0])[:10]
        plt.title(table_name + u"\n(近{0}天){1} --- {2}".format(numn, d_start, d_end), fontproperties=my_font)

        plt.grid(alpha=0.8, ls="-.")

        data = data[::-1]
        plt.xticks(range(numn), [data[i] if i % 3 == 0 else '' for i in range(numn)], rotation=45)
        miny = int(min(df['日增长率'][numn - 1::-1] * 100))
        maxy = int(max(df['日增长率'][numn - 1::-1] * 100))
        plt.yticks(range(miny - 1, maxy + 1, 1))
        plt.axhline(c='red')

        plt.xlabel(u"日期", fontproperties=my_font)
        plt.ylabel(u"日增长率", fontproperties=my_font)
        plt.legend(loc='upper left', fontsize='x-large')
        if self.con.do_figure_save:
            plt.savefig(self.con.figure_save_path + table_name + "增长天数日增长率(近{}天)_{}.png".format(numn, datetime.now().strftime('%Y%m%d')))
        return stockName

if __name__ == '__main__':
    pass