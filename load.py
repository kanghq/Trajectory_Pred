import pandas as pd
import numpy as np
import datetime

def loadFile(path):
    try:
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        print(path)
    with open(path) as f:
        for _ in range(6): #skip firs 6 lines
            next(f)
        data = pd.read_csv(f,usecols=[0,1,5,6],names = ["lat","long","date","time"], parse_dates={'datetime': ['date', 'time']},date_parser=dateparse)


        data = data.sort_values(by=['datetime'])
        data = data.reset_index(drop=True)


        diffdata = data
        '''
        diffdata = data.diff()
        diffdata.loc[0] = data.loc[0]
        diffdata.iloc[0,0] = data.loc[0][0]-datetime.datetime(1980,1,1)
        '''

        '''
        org_val = diffdata['datetime'][0]
        diffdata['datetime'] = pd.DataFrame.cumsum(diffdata['datetime'])-org_val
        diffdata['datetime'][0] = org_val
        '''
        diffdata['datetime'] = ((diffdata['datetime'] - datetime.datetime(2017,1,1)).dt.total_seconds())/86400
        diffdata['lat'] = diffdata['lat']/180
        diffdata['long']= diffdata['long']/360

        dataList = diffdata.values.tolist()
        '''
        data['datetime'] = pd.to_timedelta(data['datetime']).dt.seconds
        dataList = data.values.tolist()
        '''
        nparr = np.array(dataList,ndmin=2)
        return(nparr)


def getDestination(path):
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    with open(path) as f:
        for _ in range(6):
            next(f)
        data = pd.read_csv(f,usecols=[0,1,5,6],names = ["lat","long","date","time"], parse_dates={'datetime': ['date', 'time']},date_parser=dateparse)

        data = data.sort_values(by=['datetime'])
        data = data.reset_index(drop=True)
        return(data.tail(1))

def getRowNumber(lat,bottomBound, topBound, height,step):
    if(lat>topBound):
        return step-1
    if(lat<bottomBound):
        return 0
    return int((lat-bottomBound)/height)

def getColNumber(long,leftBound, rightBound, width,step):
    if(long>rightBound):
        return step-1
    if(long<leftBound):
        return 0
    return int((long-leftBound)/width)

def getCellNum(lat,long,leftBound,rightBound,bottomBound,topBound,width,height,step):
    col = getColNumber(long,leftBound,rightBound,width,step)
    row = getRowNumber(lat,bottomBound,topBound,height,step)
    seq = (row-1)*step+col
    if( seq <0):
        return 0
    if(seq > step*step-1):
        return step*step-1
    return seq

def df2oneHotMatrix(dataFrame,leftBound,rightBound,bottomBound,topBound,step):
    cellWidth = (rightBound-leftBound)/step
    cellHeight = (topBound-bottomBound)/step
    locationDF = dataFrame.apply(lambda x: getCellNum(x[0],x[1],leftBound,rightBound,bottomBound,topBound,cellWidth,cellHeight,step),axis=1)#.to_frame()

    return locationDF

# def convertTimeStamp2Step(dataFrame):





def oneHotEnc(num, step):
    vector = [0 for _ in range(step*step)]
    vector[num] = 1
    return vector

def loadn2HotMatrix(path,leftBound,rightBound,bottomBound,topBound,step):
    df = loadFile(path)
    return df2oneHotMatrix(df,leftBound,rightBound,bottomBound,topBound,step)

if __name__ == "__main__":
    df = loadFile('/Users/kanghuaqiang/Downloads/Geolife Trajectories 1.3/Data/181/Trajectory/20071207100605.plt')
    print(df)
    df = getDestination('/Users/kanghuaqiang/Downloads/Geolife Trajectories 1.3/Data/181/Trajectory/20071207100605.plt')
    print(df)