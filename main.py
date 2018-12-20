#coding=utf-8

from util import *
# 本程序处理协同过滤和推荐的问题
# 1.读取原始数据
# 2.格式化结构数据
# 3.处理稀疏矩阵的问题(dataset1中没有涉及到稀疏矩阵) 需要生成偏好矩阵,
# 3.1 随机生成一个用户-web服务的稀疏矩阵，因为有两个表格，因此应当生成两个稀疏矩阵，分别为rtMatrix_sprase.txt 和tpMatrix_sprise.txt。
# 3.2 根据偏好矩阵进行推荐系统的推荐运算实验。在实验数据得到后，对比参照精确度的时候使用完整的rtMatrix.txt
# 4.寻找用户之间的欧几里得距离（euclidea metric）（也称欧式距离）
# 5.计算用户表的皮尔逊相关度
# 6.计算Webservice的欧式距离相似度
# 7.计算webservice的皮尔逊相关度
# 8.使用基于用户的协同过滤(User-CF)输出推荐结果
# 8.使用基于物品的协同过滤(Item-CF)输出推荐结果
# refrence: http://python.jobbole.c`om/86563/
# refrence: https://github.com/zthero/python/blob/master/collective/code/test/make_data.py
# input Lat_A 纬度A
# input Lng_A 经度A
# input Lat_B 纬度B
# input Lng_B 经度B
# output distance 距离(km)


# 生成一个偏好矩阵
def generate_PrefMat():
    ram = random.SystemRandom()
    for u1 in UserList.keys():
        usr = UserList[u1]
        # 随机选取一些(0-1000个之间)web服务的ID然后查找出响应时间即可。
        rn = int(ram.random() * 1000)
        ln = len(WsList.keys())
        webs = {}
        for r1 in range(rn):
            # 随机选一个web服务
            l1 = int(ram.random() * ln)
            ws = WsList[str(l1)]
            # 查找出对应的响应时间属性
            rt = RTMatrix[int(usr['ID'])][int(ws['ID'])]
            webs[ws['ID']] = float(rt)
        Pref_RTMatrix[usr['ID']] = webs
        print
        usr['ID'], usr['IPAddr'], len(webs)
    saveObj(Pref_RTMatrix, GEN_RTMATRIX)
    print
    len(Pref_RTMatrix)

# 计算基于地理位置的相关度
def sim_geo(prefs, p1, p2):
    si = {}
    for itemId in prefs[p1]:
        if itemId in prefs[p2]:
            si[itemId] = 1
    # no same item
    if len(si) == 0: return 0
    SLAT1 = UserList[p1]['Latitude']
    SLON1 = UserList[p1]['Longitude']
    SLAT2 = UserList[p2]['Latitude']
    SLON2 = UserList[p2]['Longitude']
    if SLAT1.startswith("null") or SLON1.startswith("null") or SLAT2.startswith("null") or SLON2.startswith("null"):
        return 0
    LAT1 = float(SLAT1)
    LON1 = float(SLON1)
    LAT2 = float(SLAT2)
    LON2 = float(SLON2)
    # print LAT1,LON1,LAT2,LON2
    try:
        distance = getGeoDistance(LAT1, LON1, LAT2, LON2)
    except ZeroDivisionError:
        ##WTF happened?
        return 0
    except ValueError:
        return 0

    return 1000 / (distance + 1)


# 欧几里得距离
def sim_distance(prefs, p1, p2):
    si = {}
    for itemId in prefs[p1]:
        if itemId in prefs[p2]:
            si[itemId] = 1
    # no same item
    if len(si) == 0: return 0
    sum_of_squares = 0.0

    # 计算距离
    sum_of_squares = sum([pow(prefs[p1][item] - prefs[p2][item], 2) for item in si])
    return 1 / (1 + sqrt(sum_of_squares))
    pass


# 皮尔逊相关度
# 皮尔逊相关系数是一种度量两个变量间相关程度的方法。它是一个介于 1 和 -1 之间的值，其中，1 表示变量完全正相关， 0 表示无关，-1 表示完全负相关。
def sim_pearson(prefs, p1, p2):
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]: si[item] = 1
    if len(si) == 0: return 0
    n = len(si)
    # 计算开始
    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])
    sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])
    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])
    num = pSum - (sum1 * sum2 / n)
    den = math.sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
    # 计算结束
    if den == 0: return 0
    r = num / den
    return r


# 推荐用户
def topMatches(prefs, person, n=5, similarity=sim_distance):
    # python列表推导式
    scores = [(similarity(prefs, person, other), other) for other in prefs if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]


# 基于用户推荐物品
def getRecommendations(prefs, person, similarity=sim_pearson):
    totals = {}
    simSums = {}

    for other in prefs:
        # 不和自己做比较
        if other == person:
            continue
        sim = similarity(prefs, person, other)
        # 去除负相关的用户
        if sim == 0: continue
        for item in prefs[other]:
            # 只对自己没见过的服务做评价
            if item in prefs[person]: continue
            totals.setdefault(item, 0)
            totals[item] += sim * prefs[other][item]
            simSums.setdefault(item, 0)
            simSums[item] += sim
    # 归一化处理生成推荐列表
    rankings = []
    for item in totals:
        if simSums[item] <> 0:
            rankings.append((totals[item] / simSums[item], item))
    # rankings=[(total/simSums[item],item) for item,total in totals.items()]
    rankings.sort()
    rankings.reverse()
    return rankings


# 基于物品的列表
def transformPrefs(prefs):
    itemList = {}
    for person in prefs:
        for item in prefs[person]:
            if not itemList.has_key(item):
                itemList[item] = {}
                # result.setdefault(item,{})
            itemList[item][person] = prefs[person][item]
    return itemList


# 构建基于物品相似度数据集
def calculateSimilarItems(prefs, n=10):
    result = {}
    itemPrefs = transformPrefs(prefs)
    c = 0
    for item in itemPrefs:
        c += 1
        if c % 10 == 0: print
        "%d / %d" % (c, len(itemPrefs))
        scores = topMatches(itemPrefs, item, n=n, similarity=sim_distance)
        result[item] = scores
    return result


# 基于物品的推荐
def getRecommendedItems(prefs, itemMatch, user):
    userRatings = prefs[user]
    scores = {}
    totalSim = {}
    # Loop over items rated by this user
    for (item, rating) in userRatings.items():
        # Loop over items similar to this one
        for (similarity, item2) in itemMatch[item]:

            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
            # Weighted sum of rating times similarity
            scores.setdefault(item2, 0)
            scores[item2] += similarity * rating
            # Sum of all the similarities
            totalSim.setdefault(item2, 0)
            totalSim[item2] += similarity

    # Divide each total score by total weighting to get an average
    rankings = [(score / totalSim[item], item) for item, score in scores.items()]

    # Return the rankings from highest to lowest
    rankings.sort()
    rankings.reverse()
    return rankings


###API
def getRecommend(UserID, sim):
    Pref_RTMatrix = loadObjsIfExist(GEN_RTMATRIX)
    similarity = sim_pearson
    if sim == "distance":
        similarity = sim_distance
    if sim == "geo":
        similarity = sim_geo

    recomm = getRecommendations(Pref_RTMatrix, UserID, similarity)
    result = {}
    result["UserInfo"] = WsList[UserID]
    result["Recommend Services"] = []
    for im in recomm:
        # 取大于2.0的推荐值，这个值可以自己定义，如果取值太小，则表现出推荐太多无用的内容。
        if im[0] > 2.0:
            result["Recommend Services"].append(WsList[im[1]])
            # print im , WsList[im[1]]
        else:
            continue
    result["Total"] = len(result["Recommend Services"])
    print
    sim, result["Total"], result["UserInfo"]
    jresult = json.dumps(result)
    return jresult



def getAll():
    for usr in UserList.keys():
        getRecommend(usr, "pearson")
    for usr in UserList.keys():
        getRecommend(usr, "distance")
    for usr in UserList.keys():
        getRecommend(usr, "geo")


def main():
    readUserList()
    readWsList()
    readTPandRX()
    Pref_RTMatrix = loadObjsIfExist(GEN_RTMATRIX)
    if (Pref_RTMatrix is None):
        generate_PrefMat()
        ## just reload PrefMatrix
        Pref_RTMatrix = loadObjsIfExist(GEN_RTMATRIX)
    print sim_distance(Pref_RTMatrix, "101","20")
    print sim_pearson(Pref_RTMatrix,"101","20")
    result =  getRecommendations(Pref_RTMatrix,"11",sim_distance)
    print len(result)
    for im in result[0:40]:
       print im , WsList[im[1]]

    if len(sys.argv) > 1:
        getAll()

    si = []
    # checking acuuracy
    for itemId in result[0]:
        if itemId in WsList['11']:
            si[itemId] = 1

    if len(si) == 0: return 0
    sum_of_squares = 0.0

    # 计算距离
    sum_of_squares = sum([pow(prefs[p1][item] - prefs[p2][item], 2) for item in si])
    return 1 / (1 + sqrt(sum_of_squares))
    pass

    cnt = 0
    for each in inded:
        if each in h_rated[0][ip_select:]:
            cnt += 1

    doit()

    rmse = (op_to_be_pred - cntt) / op_to_be_pred

    print('RMSE , MAE = ', rmse)

    accuracy = cntt / op_to_be_pred
    print('accuracy is ', accuracy)


if __name__ == '__main__':
    main()
else:
    main()
print
__name__, "import ok!"
