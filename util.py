#coding=utf-8
import os, random, json
import pickle
from math import *
import math
import sys


# 用户字典
UserList = {}
# web服务字典
WsList = {}

# 生成的偏好矩阵
Pref_TPMatrix = {}
Pref_RTMatrix = {}

# 原始的访问数据，用来做对比
TPMatrix = []
RTMatrix = []


##原始数据
RAW_USRLIST = "./dataset1/userlist.txt"
RAW_WSLIST = "./dataset1/wslist.txt"

##参照数据
RAW_TPMATRIX = "./dataset1/tpMatrix.txt"
RAW_RTMATRIX = "./dataset1/rtMatrix.txt"

##生成的偏好矩阵
GEN_TPMATRIX = "./dataset1/tpMatrix.pref.txt"
GEN_RTMATRIX = "./dataset1/rtMatrix.pref.txt"

# 构造数据user结构
def WSUser(ID="0", IPAddr="0.0.0.0", Country="", IPNo="0", AS="0", Latitude="0", Longitude="0"):
    result = {}
    result['ID'] = ID
    result['IPAddr'] = IPAddr
    result['Country'] = Country
    result['IPNo'] = IPNo
    result['AS'] = AS
    result['Latitude'] = Latitude
    result['Longitude'] = Longitude
    return result

# 构造service的数据结构
def WSService(ID="0", WSDLAddress="", ServiceProvider="", IPAddr="0.0.0.0", Country="", IPNo="0", AS="0", Latitude="0",
              Longitude="0"):
    result = {}
    result['ID'] = ID
    # result['WSDLAddress']=WSDLAddress
    # result['ServiceProvider']=ServiceProvider
    result['IPAddr'] = IPAddr
    # result['Country']=Country
    result['IPNo'] = IPNo
    # result['AS']=AS
    result['Latitude'] = Latitude
    result['Longitude'] = Longitude
    return result

# 读取用户列表
def readUserList():
    uFile = open(RAW_USRLIST, "r")
    for line in uFile:
        if (line.startswith("[") or line.startswith("=")):
            continue
        # print line
        lns = line.split('\t')
        if (len(lns) < 6):
            print
            "Line: ", line, "has problem, skip"
            continue
        newUsr = WSUser(lns[0], lns[1], lns[2], lns[3], lns[4], lns[5], lns[6])
        UserList[lns[0]] = newUsr
    print
    "total read User:", len(UserList)


# 读取web服务列表
def readWsList():
    uFile = open(RAW_WSLIST, "r")
    for line in uFile:
        if (line.startswith("[") or line.startswith("=")):
            continue
        # print line
        lns = line.split('\t')
        if (len(lns) < 8):
            print
            "Line: ", line, "has problem, skip"
            continue
        newWs = WSService(lns[0], lns[1], lns[2], lns[3], lns[4], lns[5], lns[6], lns[7], lns[8])
        WsList[lns[0]] = newWs
    print
    "total read Webservices:", len(WsList)

def getGeoDistance(Lat_A, Lng_A, Lat_B, Lng_B):
    ra = 6378.140  # 赤道半径 (km)
    rb = 6356.755  # 极半径 (km)
    flatten = (ra - rb) / ra  # 地球扁率
    rad_lat_A = radians(Lat_A)
    rad_lng_A = radians(Lng_A)
    rad_lat_B = radians(Lat_B)
    rad_lng_B = radians(Lng_B)
    pA = atan(rb / ra * tan(rad_lat_A))
    pB = atan(rb / ra * tan(rad_lat_B))
    xx = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(rad_lng_A - rad_lng_B))
    c1 = (sin(xx) - xx) * (sin(pA) + sin(pB)) ** 2 / cos(xx / 2) ** 2
    c2 = (sin(xx) + xx) * (sin(pA) - sin(pB)) ** 2 / sin(xx / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (xx + dr)
    return distance


# 读取响应时间关系矩阵
def readTPandRX():
    uFile = open(RAW_TPMATRIX, "r")
    for line in uFile:
        if (line.startswith("[") or line.startswith("=")):
            continue
        # print line
        lns = line.split('\t')
        TPMatrix.append(lns)
    print
    "total read user-item matrix of response-time.:", len(TPMatrix), "*", len(TPMatrix[0])
    uFile = open(RAW_RTMATRIX, "r")
    for line in uFile:
        if (line.startswith("[") or line.startswith("=")):
            continue
        # print line
        lns = line.split('\t')
        RTMatrix.append(lns)
    print
    "total read user-item matrix for throughput.:", len(RTMatrix), "*", len(RTMatrix[0])

def loadObjsIfExist(filename):#启动的时候载入持久化的对象
    result= None
    if os.path.exists(filename):
        pkl_file = open(filename, 'rb')
        result = pickle.load(pkl_file)
        pkl_file.close()
    return result

def saveObj(obj,filename):#dump对象到本地
    output = open(filename, 'wb+')
    pickle.dump(obj,output)
    output.close()
