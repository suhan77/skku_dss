from pykiwoom.kiwoom import *
import datetime

class KW:

    def __init__(self):
        self.kiwoom = Kiwoom()

    def login(self):
        self.kiwoom.CommConnect(block=True)

    def getCodeList(self):
        kospiList = []
        codeList = self.kiwoom.GetCodeListByMarket('0')

        for code in codeList:
            name = self.kiwoom.GetMasterCodeName(code)
            kospiList.append((code, name))

        return kospiList

    def getConnectState(self):
        state = self.kiwoom.GetConnectState()
        if state == 0:
            return '미연결'
        elif state == 1:
            return '연결완료'

    def getStockInfo(self, code):
        df = self.kiwoom.block_request("opt10001",
                                       종목코드=code,
                                       output="주식기본정보",
                                       next=0)
        return df

    def getDailyData(self, list):
        now = datetime.datetime.now()
        today = now.strftime("%Y%m%d")
        resultList = []
        for code in list:
            df = self.kiwoom.block_request("opt10081",
                                           종목코드=code,
                                           기준일자=today,
                                           수정주가구분=1,
                                           output="주식일봉차트조회",
                                           next=0)
            resultList.append(df)

        return resultList


if __name__ == '__main__':
    kiwoom = KW()
    kiwoom.login()
    print('connectStatus : ' + kiwoom.getConnectState())

    codeList = kiwoom.getCodeList()
    print('codeList size : ' + str(len(codeList)))
    print(codeList)

    # 기본정보 삼성
    stockInfo = kiwoom.getStockInfo('005930')
    print('stockInfo ' + str(type(stockInfo)))
    print('stockInfo ' + str(stockInfo))

    # 일봉데이터 삼성, 하이닉스만 가져와본다.
    resultList = kiwoom.getDailyData(['005930', '000660'])
    print('resultList size : ' + str(len(resultList)))
    for result in resultList:
        print('stockInfo ' + str(type(result)))
        print(result.head(10))