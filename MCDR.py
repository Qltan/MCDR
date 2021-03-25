import copy
import datetime
import json
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import pydotplus
import random
import re
import time
from math import *
from sklearn import metrics

_CLUSTER_DATA = './bike_sharing_data/mydata'
RATEDATA = './bike_sharing_data/mydata/'
rateName = 'rental_return_rate_cluster_6_month_678_timedelta_5.json'
# STATION_STATUS = './station_status_by_id'


def getMonthCluster():
    cluster = '6'
    month = 678
    return month, cluster


def getCluster():
    with open(os.path.join(_CLUSTER_DATA, 'clusters.json'), 'r') as f:
        clusters = json.load(f)
        del clusters['5']['402']
        del clusters['5']['491']
    return clusters


def getRateData():
    with open(os.path.join(RATEDATA, rateName), 'r') as f:
        rateData = json.load(f)
    return rateData


def getPositionAndStations_id():
    clusters = getCluster()
    month, cluster = getMonthCluster()
    use_cluster = clusters[cluster]
    stations_id = []
    position = {}
    for key, values in use_cluster.items():
        stations_id.append(key)
        position[key] = values['position']
    return position, stations_id


def getInitialInfo():
    month, cluster = getMonthCluster()
    pattern2 = re.compile('^cluster_[0-9]+_')
    filelist2 = os.listdir(_CLUSTER_DATA)
    for filename in filelist2:
        if filename == 'cluster_6_month_678_initialStationInfo.json':
            cluster1 = filename.split('_')[1]
            month1 = filename.split('_')[3]
            if cluster1 == str(cluster) and month1 == str(month):
                print(filename)
                with open(os.path.join(_CLUSTER_DATA, filename), 'r') as f:
                    initialInfo = json.load(f)
    return initialInfo


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # print lat1, lng1, lat2, lng2
    a = lat1 - lat2
    b = lng1 - lng2
    s = 2 * asin(sqrt(pow(sin(a / 2), 2) + cos(lat1) * cos(lat2) * pow(sin(b / 2), 2)))
    earth_radius = 6378.137
    s = s * earth_radius
    if s < 0:
        return round(-s, 3)
    else:

        return round(s, 3)
    return h


def manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b


def getNeighbor(stations_id, position):
    neighbor = {}
    maxDis = 0
    for station_id1 in stations_id:
        nei = []

        for station_id2 in stations_id:
            d = manhattan_distance(position[str(station_id1)][0], position[str(station_id1)][1],
                                   position[str(station_id2)][0], position[str(station_id2)][1])
            if 0.6 > d > 0:
                nei.append(str(station_id2))
            if d > maxDis:
                maxDis = d
        neighbor[str(station_id1)] = nei
    return neighbor


def getOlderNeighbor(stations_id, position):
    neighbor = {}
    maxDis = 0
    for station_id1 in stations_id:
        nei = []

        for station_id2 in stations_id:
            d = manhattan_distance(position[str(station_id1)][0], position[str(station_id1)][1],
                                   position[str(station_id2)][0], position[str(station_id2)][1])
            if 0.9 > d > 0:
                nei.append(str(station_id2))
            if d > maxDis:
                maxDis = d
        neighbor[str(station_id1)] = nei
    return neighbor


def getMonthDayAndHour():  # month, day and hour used in this experiment
    mon = 8

    day = 99

    hour = 7

    return mon, day, hour


def getStation_status():
    monDay = {'6': 30, '7': 31, '8': 31}
    mon, day, hour = getMonthDayAndHour()
    initialByDay = {}
    totalDocksDict = {}
    initialInfo = getInitialInfo()
    position, stations_id = getPositionAndStations_id()
    for station_id, values in initialInfo.items():
        totD = values['totalDocks']
        totalDocksDict[str(station_id)] = totD

    for day in range(0, monDay[str(mon)]):
        sta = {}
        for station_id, values in initialInfo.items():
            inf = values['info']
            monInf = inf[str(mon)]
            sta[str(station_id)] = monInf[day]
        initialByDay[str(day + 1)] = sta

    station_status = {}

    for day in range(0, monDay[str(mon)]):
        station_status1 = {}
        for station_id in stations_id:
            stationInf = initialByDay[str(day + 1)][str(station_id)][str(day + 1)][str(hour)]
            station_status1[str(station_id)] = stationInf
        station_status[str(day + 1)] = station_status1
    return station_status, totalDocksDict


###########################

# MCTS algorithm
class BikeSystem(object):
    def __init__(self, availStations=[]):
        self.availStations = copy.deepcopy(availStations)

    def update(self, station_id):
        self.availStations.remove(str(station_id))


class MCTS(object):
    def __init__(self, availStations, time=6, max_actions=1000):
        self.availStations = availStations
        self.calculation_time = float(time)
        self.max_actions = max_actions

        self.confident = 8
        self.equivalence = 10000  # calc beta
        self.max_depth = 1
        self.fileCount = 0

    def get_action(self, rootStationId, starttime, neighbor, rateData, station_status, totalDocksDict, day,
                   olderNeighbor):  # rootStationId: current truck parking station
        position, stations_id = getPositionAndStations_id()
        if len(self.availStations) == 1:
            return self.availStations[0]
        self.visited_times = {}  # key: station_id, value: visited times
        simulations = 0
        begin = time.time()
        Q = {str(sta_id): -99999 for sta_id in self.availStations}  # recalculation Q value
        balanceBikeNums = {str(sta_id): 0 for sta_id in self.availStations}
        countmax = 0
        count = 0
        expandStaSet = set()
        # self.fileCount = 0
        while simulations < self.max_actions + 1:
            availStations_copy = copy.deepcopy(self.availStations)
            countmax, count = self.run_simulation(availStations_copy, rootStationId, Q, starttime, balanceBikeNums,
                                                  neighbor,
                                                  simulations,
                                                  expandStaSet, countmax, count, rateData, station_status,
                                                  totalDocksDict, day, olderNeighbor, position)
            simulations += 1
        # select the station with the maximum Q value
        maxQ, selected_station_id = self.select_one_station(Q, starttime, rateData, totalDocksDict, station_status, day,
                                                            rootStationId)
        print("total simulations=", simulations)
        print("Time spent in the simulation process:", str(time.time() - begin))
        print('Maximum number of access to uct:' + str(countmax))
        print('Total number of access to uct:' + str(count))
        print('Maximum Q:', maxQ)
        print('Maximum depth searched:', self.max_depth)

        return selected_station_id

    def select_one_station(self, Q, starttime, rateData, totalDocksDict, station_status, day, rootStationId):

        notInServiceLevalStas = []
        t_interval = starttime / 5
        mon = 8
        hour = 7
        month = str(mon) if int(mon) >= 10 else '0' + str(mon)
        day1 = str(day) if int(day) >= 10 else '0' + str(day)
        date = '2017-' + str(month) + '-' + str(day1)
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        for sta in self.availStations:
            rateDict = rateData[str(sta)]
            if date.weekday() < 5:
                rental_rate_0 = rateDict['rental_rate_0']
                return_rate_0 = rateDict['return_rate_0']
            elif date.weekday() < 7:
                rental_rate_0 = rateDict['rental_rate_1']
                return_rate_0 = rateDict['return_rate_1']
            availableBikes = station_status[str(day)][str(sta)]['availableBikes']
            availableDocks = station_status[str(day)][str(sta)]['availableDocks']
            totalDocks = totalDocksDict[str(sta)]
            for i in np.arange(0,
                               int(t_interval)):  # real-time bikes docks
                deltaNum = rental_rate_0[i] - return_rate_0[i]

                if deltaNum > 0:
                    availableBikes = float(availableBikes) - deltaNum
                    if availableBikes < 0:
                        availableBikes = 0
                    availableDocks = float(availableDocks) + deltaNum
                    if availableDocks > float(totalDocks):
                        availableBikes = 0
                        availableDocks = float(totalDocks)
                else:
                    availableDocks = float(availableDocks) - abs(deltaNum)
                    if availableDocks < 0:
                        availableDocks = 0
                    availableBikes = float(availableBikes) + abs(deltaNum)
                    if availableBikes > float(totalDocks):
                        availableDocks = 0
                        availableBikes = float(totalDocks)

            realBikes = availableBikes
            realDocks = availableDocks

            totalDocks = totalDocksDict[str(sta)]
            serviceLevel = []
            for docks in range(1, int(totalDocks)):
                availableBikes = int(totalDocks) - docks
                availableDocks = docks
                flag = 0
                for j in np.arange(int(t_interval), int(t_interval) + 24):
                    if j >= 48:
                        break
                    else:
                        deltaNum = rental_rate_0[j] - return_rate_0[j]
                    if deltaNum > 0:
                        availableBikes = float(availableBikes) - deltaNum
                        if availableBikes <= 1:
                            flag = 1
                            break
                        availableDocks = float(availableDocks) + deltaNum
                        if availableDocks >= float(totalDocks) - 1:
                            flag = 1
                            break
                    else:
                        availableDocks = float(availableDocks) - abs(deltaNum)
                        if availableDocks <= 1:
                            flag = 1
                            break
                        availableBikes = float(availableBikes) + abs(deltaNum)
                        if availableBikes >= float(totalDocks) - 1:
                            flag = 1
                            break
                if flag == 0:
                    serviceLevel.append(int(totalDocks) - int(docks))
            if serviceLevel == [] or float(realBikes) < min(serviceLevel) or float(realBikes) > max(serviceLevel):
                notInServiceLevalStas.append(sta)

        if not notInServiceLevalStas:
            maxQ, sta_id = max((Q[str(sta_id)], sta_id) for sta_id in self.availStations)
        else:
            maxQ, sta_id = max((Q[str(sta_id)], sta_id) for sta_id in notInServiceLevalStas)
            if maxQ == -99999:
                minDis = 10000
                print(notInServiceLevalStas)
                position, stations_id = getPositionAndStations_id()
                for sta in notInServiceLevalStas:
                    dis = manhattan_distance(position[str(rootStationId)][0], position[str(rootStationId)][1],
                                             position[str(sta)][0], position[str(sta)][1])
                    if dis < minDis:
                        minDis = dis
                        sta_id = sta
        # maxQ, sta_id = max((Q[str(sta_id)], sta_id) for sta_id in self.availStations)
        if sta_id == '238':
            print(Q)
            print('Q[238]:' + str(Q['238']))
        return maxQ, sta_id

    def run_simulation(self, availStations, rootStationId,
                       Q, starttime, balanceBikeNums, neighbor, simulations, expandStaSet, countmax,
                       count2, rateData, station_status, totalDocksDict, day, olderNeighbor,
                       position):  # conduct run_simulation and get a path
        """
        MCTS main process
        """
        visited_times = self.visited_times
        # availStations = bikeSystem.availStations
        visited_paths = []
        cumulativeDis = []  # The total travel distance of the truck
        expand = True
        selectedSta = rootStationId
        dropNum = 0
        pickNum = 0
        # simulation
        count = 0
        countRequestFlag = 0
        neiStaQ = []
        for t in range(1, self.max_actions + 1):
            lastStation = selectedSta
            if all(visited_times.get(station_id) for station_id in availStations):  # UCB
                log_total = log(sum(visited_times[str(sta_id)] for sta_id in availStations))
                value, sta_id = max((
                                        Q[str(sta_id)] + sqrt(self.confident * log_total / visited_times[str(sta_id)]),
                                        sta_id)
                                    for sta_id in
                                    availStations)
                selectedSta = sta_id
                count += 1
                count2 += 1

            else:
                availNeighbor = [sta_id for sta_id in neighbor[str(lastStation)] if sta_id in availStations]
                if len(availNeighbor) and random.random() < 0:
                    selectedSta = random.choice(availNeighbor)
                else:
                    selectedSta = random.choice(availStations)
            # bikeSystem.update(selectedSta)
            availStations.remove(str(selectedSta))
            # Expand
            if expand is True and str(selectedSta) not in visited_times:
                expand = False
                visited_times[str(selectedSta)] = 0
                expandStaSet.add(str(selectedSta))

            if t > self.max_depth:
                self.max_depth = t

            visited_paths.append(selectedSta)
            is_full = not len(availStations)
            isRequest, endtime, dropNum0, pickNum0, real_bikes, real_docks = self.getRequest(lastStation, selectedSta,
                                                                                             Q, starttime,
                                                                                             cumulativeDis, rateData,
                                                                                             station_status,
                                                                                             totalDocksDict, day,
                                                                                             position)
            starttime = endtime
            if isRequest:
                availselectedStaNeighbor = [sta_id for sta_id in olderNeighbor[str(selectedSta)] if
                                            sta_id in availStations]
                # neiStaQ = {str(sta):0 for sta in availselectedStaNeighbor}
                for neiSta in availselectedStaNeighbor:
                    cumulativeDisCopy = copy.deepcopy(cumulativeDis)
                    diss = []
                    dis = manhattan_distance(position[str(selectedSta)][0], position[str(selectedSta)][1],
                                             position[str(neiSta)][0], position[str(neiSta)][1])
                    # cumulativeDisCopy.append(dis)
                    cumulativeDisCopy.append(dis)
                    v = 7  # 10m/s  ==  36km/h  truck speed
                    t = dis * 1000 / v
                    t_arrive = starttime + round(t / 60)
                    t_interval = round(t_arrive / 5)
                    serviceLevel, real_bikess, real_dockss = self.getServiceLevel(neiSta, t_interval,
                                                                                  rateData, station_status,
                                                                                  totalDocksDict, day)
                    dropNum = 0
                    pickNum = 0
                    if not serviceLevel:  # return>>rental
                        pickNum = real_bikes
                    else:
                        minBikes = min(serviceLevel)
                        maxBikes = max(serviceLevel)

                    if minBikes <= real_bikes <= maxBikes:
                        pass
                    else:
                        if real_bikes < minBikes:
                            dropNum = minBikes - real_bikes  # TN
                        if real_bikes > maxBikes:
                            pickNum = real_bikes - maxBikes
                    balanceBikeNumss = dropNum + pickNum
                    flag = -1
                    if dropNum > 0:
                        flag = 0
                    elif pickNum > 0:
                        flag = 1
                    neiStaQ.append(self.getScore(cumulativeDisCopy, balanceBikeNumss, real_bikess, real_dockss, flag))
            if is_full or isRequest:
                break
        if count > countmax:
            countmax = count
        # Back-propagation
        balanceBikeNums[str(selectedSta)] = dropNum0 + pickNum0

        flag = -1
        if dropNum0 > 0:
            flag = 0
        elif pickNum0 > 0:
            flag = 1
        # if selectedSta=='229':
        #   print('real_docks:'+str(real_docks))
        for sta_id in visited_paths:
            if sta_id not in visited_times:
                continue
            visited_times[str(sta_id)] += 1
            if isRequest:
                if not neiStaQ:
                    neiStaQ.append(0)
                score = self.getScore(cumulativeDis, balanceBikeNums[str(selectedSta)],
                                      real_bikes, real_docks, flag) + np.mean(neiStaQ)
                Q[str(sta_id)] = (abs(Q[str(sta_id)]) * (visited_times[str(sta_id)] - 1) +
                                  score) / visited_times[str(sta_id)]
                Q[str(sta_id)] = round(Q[str(sta_id)], 4)

        log_dir = './bike_sharing_data/mydata/log/' + str(self.fileCount + 1)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(log_dir + '/' + str(simulations + 1) + '.json',
                  'w') as f:
            json.dump(Q, f)

        return countmax, count2

    def getScore(self, cumulativeDis, balanceNums, real_bikes, real_docks, flag):
        disScore = 0
        numScore = 0
        score = 0
        if sum(cumulativeDis) <= 300:
            disScore = 10
        elif sum(cumulativeDis) <= 600:
            disScore = 6
        elif sum(cumulativeDis) <= 1000:
            disScore = 4
        elif sum(cumulativeDis) <= 1500:
            disScore = 2
        elif sum(cumulativeDis) <= 2000:
            disScore = 0
        else:
            disScore = -5
        # dis = sum(cumulativeDis)
        # if dis>=3000:
        #   disScore = -10
        # elif dis>=2000:
        #   disScore = 20-10*(dis/1000)
        # elif dis>=0:
        #   disScore = 10-5*(dis/1000)

        if balanceNums == 0:
            numScore = 0
        elif balanceNums <= 3:
            numScore = 2
        elif balanceNums <= 6:
            numScore = 4
        elif balanceNums <= 10:
            numScore = 6
        else:
            numScore = 10
        # if balanceNums >=10:
        #   numScore = 10
        # else:
        #   numScore = balanceNums

        urgencyScore = 0
        if flag == 0 and real_bikes <= 1:
            urgencyScore = 10
        elif flag == 1 and real_docks <= 1:
            urgencyScore = 10
        elif flag == -1:
            return 0
        score = 0.5 * disScore + 0.5 * numScore + urgencyScore
        return score

    def getRequest(self, lastStation, selectedSta, Q, starttime, cumulativeDis, rateData, station_status,
                   totalDocksDict, day, position):
        dis = manhattan_distance(position[str(lastStation)][0], position[str(lastStation)][1],
                                 position[str(selectedSta)][0], position[str(selectedSta)][1])
        cumulativeDis.append(round(dis * 1000, 3))
        noise = abs(np.random.normal(loc=0.0, scale=2))
        v = 7  # 8m/s  ==  36km/h
        t = dis * 1000 / v  #
        t_arrive = starttime + round(t / 60)
        t_interval = round(t_arrive / 5)

        serviceLevel, real_bikes, real_docks = self.getServiceLevel(selectedSta, t_interval, rateData, station_status,
                                                                    totalDocksDict, day)
        dropNum = 0
        pickNum = 0
        endtime = t_arrive
        if not serviceLevel:  # return>>rental
            endtime = t_arrive + real_bikes * 0.3 + noise
            pickNum = real_bikes
            return True, endtime, dropNum, pickNum, real_bikes, real_docks
        else:
            minBikes = min(serviceLevel)
            maxBikes = max(serviceLevel)

        if minBikes <= real_bikes <= maxBikes:
            endtime = t_arrive + noise
            return False, endtime, dropNum, pickNum, real_bikes, real_docks
        else:
            if real_bikes < minBikes:
                dropNum = minBikes - real_bikes
                endtime = t_arrive + dropNum * 0.3 + noise  # drop/take time (30s)
            if real_bikes > maxBikes:
                pickNum = real_bikes - maxBikes
                endtime = t_arrive + pickNum * 0.3 + noise
            return True, endtime, dropNum, pickNum, real_bikes, real_docks

    def getServiceLevel(self, selectedSta, t_interval, rateData, station_status, totalDocksDict, day):
        # mon,day,hour = getMonthDayAndHour()
        mon = 8
        hour = 7
        rateDict = rateData[str(selectedSta)]
        t_intervalFlag = 0
        if hour == 7:
            t_intervalFlag = 0
        elif hour == 8:
            t_intervalFlag = 12
        elif hour == 9:
            t_intervalFlag = 24
        month = str(mon) if int(mon) >= 10 else '0' + str(mon)
        day1 = str(day) if int(day) >= 10 else '0' + str(day)
        date = '2017-' + str(month) + '-' + str(day1)
        date = datetime.datetime.strptime(date, '%Y-%m-%d')

        if date.weekday() < 5:
            rental_rate_0 = rateDict['rental_rate_0']
            return_rate_0 = rateDict['return_rate_0']
        elif date.weekday() < 7:
            rental_rate_0 = rateDict['rental_rate_1']
            return_rate_0 = rateDict['return_rate_1']
        iniBikes = station_status[str(day)][str(selectedSta)]['availableBikes']
        iniDocks = station_status[str(day)][str(selectedSta)]['availableDocks']
        totalDocks = totalDocksDict[str(selectedSta)]
        serviceLevel = []
        availableBikes = iniBikes
        # print('selectedSta:'+str(selectedSta))
        availableDocks = iniDocks
        # print('selectedSta:'+str(selectedSta))
        # print('iniBikes:'+str(iniBikes))
        for i in np.arange(int(t_intervalFlag), int(t_interval) + int(t_intervalFlag)):  # real-time bikes docks
            deltaNum = 0
            deltaNum = rental_rate_0[i] - return_rate_0[i]

            if float(availableBikes) < 1.0:
                pass  # rental_lost += deltNum
            if float(availableDocks) < 1.0:
                pass  # return_lost += deltNum

            if deltaNum > 0:
                availableBikes = float(availableBikes) - deltaNum
                if availableBikes < 0:
                    availableBikes = 0
                availableDocks = float(availableDocks) + deltaNum
                if availableDocks > float(totalDocks):
                    availableBikes = 0
                    availableDocks = float(totalDocks)
            else:
                availableDocks = float(availableDocks) - abs(deltaNum)
                if availableDocks < 0:
                    availableDocks = 0
                availableBikes = float(availableBikes) + abs(deltaNum)
                if availableBikes > float(totalDocks):
                    availableDocks = 0
                    availableBikes = float(totalDocks)

        realBikes = availableBikes
        realDocks = availableDocks

        for docks in range(1, int(totalDocks)):
            availableBikes = int(totalDocks) - docks
            availableDocks = docks
            flag = 0
            for j in np.arange(int(t_intervalFlag) + int(t_interval), int(t_interval) + int(t_intervalFlag) + 24):

                deltaNum = 0
                if j >= 48:
                    break
                else:
                    try:
                        deltaNum = rental_rate_0[j] - return_rate_0[j]
                    except:
                        print('raredata error! j:' + str(j))
                if deltaNum > 0:
                    availableBikes = float(availableBikes) - deltaNum
                    if availableBikes <= 1:
                        flag = 1
                        # print('availableBikes:'+str(availableBikes))
                        break
                    availableDocks = float(availableDocks) + deltaNum
                    if availableDocks >= float(totalDocks) - 1:
                        flag = 1
                        # print('availaableDocks:'+str(availableDocks))
                        break
                else:
                    availableDocks = float(availableDocks) - abs(deltaNum)
                    if availableDocks <= 1:
                        # print('availableDocks:'+str(availableDocks))
                        flag = 1
                        break
                    availableBikes = float(availableBikes) + abs(deltaNum)
                    if availableBikes >= float(totalDocks) - 1:
                        # print('availableBikes:'+str(availableBikes))
                        flag = 1
                        break
            if flag == 0:
                serviceLevel.append(int(totalDocks) - int(docks))

        return serviceLevel, math.floor(float(realBikes)), math.floor(float(realDocks))


def start(availStations, neighbor, lostNums, visitedPath, cumulativeDis, startStation, balanceNum, mutex, realtimeBikes,
          day, olderNeighbor):
    print("start running, the process number is %d" % (os.getpid()))
    mcts = MCTS(availStations)
    selectedSta = startStation
    starttime = 0
    rateData = getRateData()
    station_status, totalDocksDict = getStation_status()
    # visitedPath = []
    # cumulativeDis = []
    info = {}
    visitedPath.append(selectedSta)
    totalLost = 0
    print('start station:' + str(selectedSta))
    # lostNums = {}
    isRequest, starttime, dropNum, pickNum, rentalLost, returnLost, realbikes = getRequest(selectedSta, selectedSta,
                                                                                           starttime,
                                                                                           cumulativeDis, rateData,
                                                                                           station_status,
                                                                                           totalDocksDict, day)
    lostNums[str(selectedSta)] = float(rentalLost) + float(returnLost)
    totalLost += lostNums[str(selectedSta)]
    info['time'] = starttime
    info['realbikes'] = realbikes
    realtimeBikes[str(selectedSta)] = info

    if int(dropNum) > 0:
        balanceNum[str(selectedSta)] = -int(dropNum)
    elif int(pickNum) > 0:
        balanceNum[str(selectedSta)] = int(pickNum)
    else:
        balanceNum[str(selectedSta)] = 0
    if isRequest:
        print('sub-process:pid=%d' % os.getpid())
        print('balance station:' + str(selectedSta) + ' dropNum:' + str(dropNum) + ' pickNum:' + str(pickNum))
        print('customer loss:' + str(lostNums[str(selectedSta)]))
        print('current time:' + str(starttime) + ' min')
        print('travel distance:')
        print(cumulativeDis)
        # bikeSystem.update(selectedSta)
        availStations.remove(str(selectedSta))
    mcts.fileCount = 0
    while 1:
        lastSta = selectedSta
        info = {}
        mutex.acquire()
        if not len(availStations):
            print('There are no stations need to be balanced')
            lostNums['totalLost'] = totalLost
            mutex.release()
            break
        selectedSta = mcts.get_action(lastSta, starttime, neighbor, rateData, station_status, totalDocksDict, day,
                                      olderNeighbor)
        mcts.fileCount += 1
        print('through station:' + str(selectedSta))
        # bikeSystem.update(selectedSta)
        availStations.remove(str(selectedSta))
        mutex.release()

        visitedPath.append(selectedSta)

        isRequest, starttime, dropNum, pickNum, rentalLost, returnLost, realbikes = getRequest(lastSta, selectedSta,
                                                                                               starttime,
                                                                                               cumulativeDis, rateData,
                                                                                               station_status,
                                                                                               totalDocksDict, day)
        lostNums[str(selectedSta)] = float(rentalLost) + float(returnLost)
        totalLost += lostNums[str(selectedSta)]
        info['time'] = starttime
        info['realbikes'] = realbikes
        realtimeBikes[str(selectedSta)] = info

        if int(dropNum) > 0:
            balanceNum[str(selectedSta)] = -int(dropNum)
        elif int(pickNum) > 0:
            balanceNum[str(selectedSta)] = int(pickNum)
        else:
            balanceNum[str(selectedSta)] = 0

        if isRequest:
            print('sub-process:pid=%d' % os.getpid())
            print('balance station:' + str(selectedSta) + ' dropNum:' + str(dropNum) + ' pickNum:' + str(pickNum))
            print('customer loss:' + str(lostNums[str(selectedSta)]))
            print('current time:' + str(starttime) + ' min')
            print('travel distance:')
            print(cumulativeDis)
        if not len(availStations):
            print('There are no stations need to be balanced')
            lostNums['totalLost'] = totalLost
            break
        print('****************************************************')


def getRequest(lastStation, selectedSta, starttime, cumulativeDis, rateData, station_status, totalDocksDict, day):
    position, stations_id = getPositionAndStations_id()
    dis = manhattan_distance(position[str(lastStation)][0], position[str(lastStation)][1],
                             position[str(selectedSta)][0],
                             position[str(selectedSta)][1])
    cumulativeDis.append(round(dis * 1000, 3))
    noise = abs(abs(np.random.normal(loc=0.0, scale=2)))
    v = 7  # 10m/s  ==  36km/h
    t = dis * 1000 / v
    t_arrive = starttime + t // 60
    t_interval = t_arrive // 5
    dropNum = 0
    pickNum = 0
    realbikes = 0
    serviceLevel, real_bikes, real_docks, rentalLost, returnLost = getServiceLevel(selectedSta, t_interval, rateData,
                                                                                   station_status, totalDocksDict, day)
    if not serviceLevel:  # return>>rental
        print('serviceLevel is null')
        endtime = t_arrive + real_bikes * 0.3 + noise
        pickNum = real_bikes
        realbikes = 0
        return True, endtime, dropNum, pickNum, realbikes, returnLost, realbikes
    else:
        minBikes = min(serviceLevel)
        maxBikes = max(serviceLevel)

    endtime = t_arrive
    if minBikes <= real_bikes <= maxBikes:
        endtime = t_arrive + noise
        if selectedSta == '127':
            print('dropNum:' + str(dropNum))
            print('pickNum:' + str(pickNum))
        realbikes = real_bikes
        return False, endtime, dropNum, pickNum, rentalLost, returnLost, realbikes
    else:
        if real_bikes < minBikes:
            dropNum = minBikes - real_bikes
            endtime = t_arrive + dropNum * 0.3 + noise
        if real_bikes > maxBikes:
            pickNum = real_bikes - maxBikes
            endtime = t_arrive + pickNum * 0.3 + noise
        if selectedSta == '127':
            print('dropNum:' + str(dropNum))
            print('pickNum:' + str(pickNum))
        if pickNum != 0:
            realbikes = maxBikes
        elif dropNum != 0:
            realbikes = minBikes
        return True, endtime, dropNum, pickNum, rentalLost, returnLost, realbikes


def getServiceLevel(selectedSta, t_interval, rateData, station_status, totalDocksDict, day):
    # mon,day,hour = getMonthDayAndHour()
    mon = 8
    hour = 7
    rateDict = rateData[str(selectedSta)]
    t_intervalFlag = 0
    if hour == 7:
        t_intervalFlag = 0
    elif hour == 8:
        t_intervalFlag = 12
    elif hour == 9:
        t_intervalFlag = 24

    month = str(mon) if int(mon) >= 10 else '0' + str(mon)
    day1 = str(day) if int(day) >= 10 else '0' + str(day)
    date = '2017-' + str(month) + '-' + str(day1)
    date = datetime.datetime.strptime(date, '%Y-%m-%d')
    if date.weekday() < 5:
        rental_rate_0 = rateDict['rental_rate_0']
        return_rate_0 = rateDict['return_rate_0']
    elif date.weekday() < 7:
        rental_rate_0 = rateDict['rental_rate_1']
        return_rate_0 = rateDict['return_rate_1']
    iniBikes = station_status[str(day)][str(selectedSta)]['availableBikes']
    iniDocks = station_status[str(day)][str(selectedSta)]['availableDocks']
    totalDocks = totalDocksDict[str(selectedSta)]
    serviceLevel = []
    availableBikes = iniBikes
    availableDocks = iniDocks
    if selectedSta == '127':
        print('iniBikes:' + str(availableBikes))
        print('iniDocks:' + str(availableDocks))
        print('t_interval:' + str(t_interval))
    print(totalDocks)
    rentalLost = 0
    returnLost = 0
    for i in np.arange(int(t_intervalFlag), int(t_interval) + int(t_intervalFlag)):  # real-time bikes docks
        deltaNum = 0
        deltaNum = rental_rate_0[i] - return_rate_0[i]

        if float(availableBikes) < 1.0 and deltaNum > 0:
            rentalLost += deltaNum
            pass  # rental_lost += deltNum
        if float(availableDocks) < 1.0 and deltaNum < 0:
            returnLost += abs(deltaNum)
            pass  # return_lost += deltNum

        if deltaNum > 0:
            availableBikes = float(availableBikes) - deltaNum
            if availableBikes < 0:
                availableBikes = 0
            availableDocks = float(availableDocks) + deltaNum
            if availableDocks > float(totalDocks):
                availableBikes = 0
                availableDocks = float(totalDocks)
        else:
            availableDocks = float(availableDocks) - abs(deltaNum)
            if availableDocks < 0:
                availableDocks = 0
            availableBikes = float(availableBikes) + abs(deltaNum)
            if availableBikes > float(totalDocks):
                availableDocks = 0
                availableBikes = float(totalDocks)

    if selectedSta == '127':
        print('realBikes:' + str(availableBikes))
        print('realDocks:' + str(availableDocks))
    realBikes = availableBikes
    realDocks = availableDocks
    for docks in range(1, int(totalDocks)):
        availableBikes = int(totalDocks) - docks
        availableDocks = docks
        flag = 0
        for j in np.arange(int(t_intervalFlag) + int(t_interval), int(t_interval) + int(t_intervalFlag) + 24):
            deltaNum = 0
            if j >= 48:
                break
            else:
                deltaNum = rental_rate_0[j] - return_rate_0[j]
            if deltaNum > 0:
                availableBikes = float(availableBikes) - deltaNum
                if availableBikes <= 1:
                    flag = 1
                    # print('availableBikes:'+str(availableBikes))
                    break
                availableDocks = float(availableDocks) + deltaNum
                if availableDocks >= float(totalDocks) - 1:
                    flag = 1
                    # print('availableDocks:'+str(availableDocks))
                    break
            else:
                availableDocks = float(availableDocks) - abs(deltaNum)
                if availableDocks <= 1:
                    # print('availableDocks:'+str(availableDocks))
                    flag = 1
                    break
                availableBikes = float(availableBikes) + abs(deltaNum)
                if availableBikes >= float(totalDocks) - 1:
                    # print('availableBikes:'+str(availableBikes))
                    flag = 1
                    break
        if flag == 0:
            serviceLevel.append(int(totalDocks) - int(docks))
    if selectedSta == '127':
        print(serviceLevel)
    return serviceLevel, math.floor(float(realBikes)), math.floor(float(realDocks)), rentalLost, returnLost


def mctsAlgorithm():
    experiment_path = './bike_sharing_data/mydata/experiment_result2'
    # month, day, hour = getMonthDayAndHour()
    month = 8
    hour = 7
    day1 = [i for i in range(1, 32)]
    day2 = [5, 6, 12, 13, 19, 20, 26, 27]  # The weekend of August!
    days = [i for i in day1 if i not in day2]

    #  11 -> 1
    for day in days:
        position, stations_id = getPositionAndStations_id()
        availStations = stations_id
        availStations = multiprocessing.Manager().list(availStations)
        realtimeBikes = multiprocessing.Manager().dict()

        lostNums1 = multiprocessing.Manager().dict()
        visitedPath1 = multiprocessing.Manager().list()
        cumulativeDis1 = multiprocessing.Manager().list()
        balanceNum1 = multiprocessing.Manager().dict()
        lostNums2 = multiprocessing.Manager().dict()
        visitedPath2 = multiprocessing.Manager().list()
        cumulativeDis2 = multiprocessing.Manager().list()
        balanceNum2 = multiprocessing.Manager().dict()

        neighbor = getNeighbor(stations_id, position)
        olderNeighbor = getOlderNeighbor(stations_id, position)
        startStation1 = '237'
        startStation2 = '369'

        mutex = multiprocessing.Lock()
        p1 = multiprocessing.Process(target=start, args=(
            availStations, neighbor, lostNums1, visitedPath1, cumulativeDis1, startStation1, balanceNum1, mutex,
            realtimeBikes, day, olderNeighbor))
        p2 = multiprocessing.Process(target=start, args=(
            availStations, neighbor, lostNums2, visitedPath2, cumulativeDis2, startStation2, balanceNum2, mutex,
            realtimeBikes, day, olderNeighbor))

        p1.start()
        p2.start()
        p1.join()
        p2.join()

        print('customer loss:' + str(lostNums1))
        print('through station:' + str(visitedPath1))
        print('balanced number:' + str(balanceNum1))
        print('travel distance:' + str(cumulativeDis1))

        print('customer loss:' + str(lostNums2))
        print('through station:' + str(visitedPath2))
        print('balanced number:' + str(balanceNum2))
        print('travel distance:' + str(cumulativeDis2))

        print('pre-process:pid=%d' % os.getpid())
        print('real status of stations:' + str(realtimeBikes))
        filename = 'result_month_' + str(month) + '_day_' + str(day) + '_hour_' + str(hour) + '.json'
        realtimeBikes1 = {}
        for sta, dicts in realtimeBikes.items():
            realtimeBikes1[str(sta)] = dicts
        experimentResult = {}
        resultTruck1 = {}
        resultTruck2 = {}

        lostNums11 = {}
        balanceNum11 = {}
        for sta, num in lostNums1.items():
            lostNums11[str(sta)] = num
        for sta, num in balanceNum1.items():
            balanceNum11[str(sta)] = num

        resultTruck1['lostUsers'] = lostNums11
        resultTruck1['visitedPath'] = list(visitedPath1)
        resultTruck1['balanceNum'] = balanceNum11
        resultTruck1['travelDis'] = list(cumulativeDis1)

        lostNums22 = {}
        balanceNum22 = {}
        for sta, num in lostNums2.items():
            lostNums22[str(sta)] = num
        for sta, num in balanceNum2.items():
            balanceNum22[str(sta)] = num

        resultTruck2['lostUsers'] = lostNums22
        resultTruck2['visitedPath'] = list(visitedPath2)
        resultTruck2['balanceNum'] = balanceNum22
        resultTruck2['travelDis'] = list(cumulativeDis2)

        experimentResult['truck1'] = resultTruck1
        experimentResult['truck2'] = resultTruck2
        experimentResult['afterBalanceRealBikes'] = realtimeBikes1

        experiment_path = './bike_sharing_data/mydata/experiment_result2/epsilon_0'
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        with open(os.path.join(experiment_path, filename), 'w') as f:
            json.dump(experimentResult, f)
        print('day' + str(day) + 'finished!')


def noRepositionStart(lostNums):
    starttime = 0
    position, stations_id = getPositionAndStations_id()
    rateData = getRateData()
    station_status, totalDocksDict = getStation_status()
    # mon,day2,hour = getMonthDayAndHour()
    mon = 8

    for day in range(1, 32):
        totalLost = 0
        lost = {}
        for station_id in stations_id:
            rateDict = rateData[str(station_id)]
            month = str(mon) if int(mon) >= 10 else '0' + str(mon)
            day1 = str(day) if int(day) >= 10 else '0' + str(day)
            date = '2017-' + str(month) + '-' + str(day1)
            date = datetime.datetime.strptime(date, '%Y-%m-%d')
            if date.weekday() < 5:
                rental_rate_0 = rateDict['rental_rate_0']
                return_rate_0 = rateDict['return_rate_0']
            elif date.weekday() < 7:
                rental_rate_0 = rateDict['rental_rate_1']
                return_rate_0 = rateDict['return_rate_1']

            iniBikes = station_status[str(day)][str(station_id)]['availableBikes']
            iniDocks = station_status[str(day)][str(station_id)]['availableDocks']
            totalDocks = totalDocksDict[str(station_id)]
            availableBikes = iniBikes
            availableDocks = iniDocks

            rentalLost = 0
            returnLost = 0
            for i in np.arange(0, 48):
                deltaNum = 0
                deltaNum = rental_rate_0[i] - return_rate_0[i]

                if deltaNum > 0 and (deltaNum - float(availableBikes)) > 0:
                    rentalLost += (deltaNum - float(availableBikes))
                    pass  # rental_lost += deltNum
                if deltaNum < 0 and (abs(deltaNum) - float(availableDocks)) > 0:
                    returnLost += (abs(deltaNum) - float(availableDocks))
                    pass  # return_lost += deltNum

                if deltaNum > 0:
                    availableBikes = float(availableBikes) - deltaNum
                    if availableBikes < 0:
                        availableBikes = 0
                    availableDocks = float(availableDocks) + deltaNum
                    if availableDocks > float(totalDocks):
                        availableBikes = 0
                        availableDocks = float(totalDocks)
                else:
                    availableDocks = float(availableDocks) - abs(deltaNum)
                    if availableDocks < 0:
                        availableDocks = 0
                    availableBikes = float(availableBikes) + abs(deltaNum)
                    if availableBikes > float(totalDocks):
                        availableDocks = 0
                        availableBikes = float(totalDocks)
            lost[str(station_id)] = rentalLost + returnLost
            totalLost += lost[str(station_id)]
        lost['totalLost'] = totalLost
        print(totalLost)
        lostNums[str(day)] = lost


def noReposition():
    experiment_path = './bike_sharing_data/mydata/noReposition'
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    # month,day,hour = getMonthDayAndHour()
    month = 8
    hour = 7
    lostNums = {}
    noRepositionStart(lostNums)
    print(lostNums)
    filename = 'noRepositionLost_month_' + str(month) + '_hour_' + str(78910) + '.json'
    with open(os.path.join(experiment_path, filename), 'w') as f:
        json.dump(lostNums, f)


def staticRepositionStart(lostNums):
    position, stations_id = getPositionAndStations_id()
    rateData = getRateData()
    station_status, totalDocksDict = getStation_status()
    mon, day, hour = getMonthDayAndHour()

    for day in range(1, 32):
        totalLost = 0
        lost = {}
        for station_id in stations_id:

            rateDict = rateData[str(station_id)]
            month = str(mon) if int(mon) >= 10 else '0' + str(mon)
            day1 = str(day) if int(day) >= 10 else '0' + str(day)
            date = '2017-' + str(month) + '-' + str(day1)
            date = datetime.datetime.strptime(date, '%Y-%m-%d')
            if date.weekday() < 5:
                rental_rate_0 = rateDict['rental_rate_0']
                return_rate_0 = rateDict['return_rate_0']
            elif date.weekday() < 7:
                rental_rate_0 = rateDict['rental_rate_1']
                return_rate_0 = rateDict['return_rate_1']

            totalDocks = totalDocksDict[str(station_id)]
            serviceLevel = []
            for docks in range(1, int(totalDocks)):
                availableBikes = int(totalDocks) - docks
                availableDocks = docks
                flag = 0
                for j in np.arange(0, 19):
                    deltaNum = 0
                    deltaNum = rental_rate_0[j] - return_rate_0[j]
                    if deltaNum > 0:
                        availableBikes = float(availableBikes) - deltaNum
                        if availableBikes <= 1:
                            flag = 1
                            # print('availableBikes:'+str(availableBikes))
                            break
                        availableDocks = float(availableDocks) + deltaNum
                        if availableDocks >= float(totalDocks) - 1:
                            flag = 1
                            # print('availableDocks:'+str(availableDocks))
                            break
                    else:
                        availableDocks = float(availableDocks) - abs(deltaNum)
                        if availableDocks <= 1:
                            # print('availableDocks:'+str(availableDocks))
                            flag = 1
                            break
                        availableBikes = float(availableBikes) + abs(deltaNum)
                        if availableBikes >= float(totalDocks) - 1:
                            # print('availableBikes:'+str(availableBikes))
                            flag = 1
                            break
                if flag == 0:
                    serviceLevel.append(int(totalDocks) - int(docks))

            iniBikes = serviceLevel[random.choice(range(0, len(serviceLevel)))]
            iniDocks = int(totalDocks) - iniBikes

            availableBikes = iniBikes
            availableDocks = iniDocks
            # if station_id == '127':
            #   print('iniBikes:' + str(availableBikes))
            #   print('iniDocks:' + str(availableDocks))
            # print(totalDocks)
            rentalLost = 0
            returnLost = 0
            for i in np.arange(0, 48):  # real-time bikes docks
                deltaNum = 0
                deltaNum = rental_rate_0[i] - return_rate_0[i]

                if deltaNum > 0 and (deltaNum - float(availableBikes)) > 0:
                    rentalLost += (deltaNum - float(availableBikes))
                    pass  # rental_lost += deltNum
                if deltaNum < 0 and (abs(deltaNum) - float(availableDocks)) > 0:
                    returnLost += (abs(deltaNum) - float(availableDocks))
                    pass  # return_lost += deltNum

                if deltaNum > 0:
                    availableBikes = float(availableBikes) - deltaNum
                    if availableBikes < 0:
                        availableBikes = 0
                    availableDocks = float(availableDocks) + deltaNum
                    if availableDocks > float(totalDocks):
                        availableBikes = 0
                        availableDocks = float(totalDocks)
                else:
                    availableDocks = float(availableDocks) - abs(deltaNum)
                    if availableDocks < 0:
                        availableDocks = 0
                    availableBikes = float(availableBikes) + abs(deltaNum)
                    if availableBikes > float(totalDocks):
                        availableDocks = 0
                        availableBikes = float(totalDocks)
            lost[str(station_id)] = rentalLost + returnLost
            totalLost += lost[str(station_id)]
        lost['totalLost'] = totalLost
        print(totalLost)
        lostNums[str(day)] = lost


def staticReposition():
    experiment_path = './bike_sharing_data/mydata/staticReposition'
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    month, day, hour = getMonthDayAndHour()
    lostNums = {}
    staticRepositionStart(lostNums)
    print(lostNums)
    filename = 'staticRepositionLost_month_' + str(month) + '_hour_' + str(78910) + '.json'
    with open(os.path.join(experiment_path, filename), 'w') as f:
        json.dump(lostNums, f)


def nearestNeihborRepositionStart(startStation, availStations, mutex, realtimeBikes, day, beforeBalancedTotalLost):
    position, stations_id = getPositionAndStations_id()
    rateData = getRateData()
    station_status, totalDocksDict = getStation_status()
    # mon, day, hour = getMonthDayAndHour()
    mon = 8
    hour = 7
    dropStation = []
    pickStation = []
    balanceStas = []

    for sta in availStations:
        iniBikes = station_status[str(day)][str(sta)]['availableBikes']
        iniDocks = station_status[str(day)][str(sta)]['availableDocks']

        if int(iniBikes) < 5:
            dropStation.append(str(sta))
            balanceStas.append(str(sta))
        if int(iniDocks) < 5:
            pickStation.append(str(sta))
            balanceStas.append(str(sta))
    # balanceSta = startStation

    starttime = 0
    v = 7
    while True:

        if starttime > 80:
            break
        info = {}
        diss = []
        minDis = 10
        pickNum = 0
        dropNum = 0
        print('balanceStas' + str(balanceStas))
        if not balanceStas:
            break
        mutex.acquire()
        balanceStas = [s for s in balanceStas if s in availStations]
        if not balanceStas:
            break
        for sta in balanceStas:
            dis = manhattan_distance(position[str(startStation)][0], position[str(startStation)][1], position[sta][0],
                                     position[sta][1])
            if dis < minDis:
                minDis = dis
                balanceSta = sta
        startStation = balanceSta
        availStations.remove(str(balanceSta))
        mutex.release()

        rateDict = rateData[str(balanceSta)]
        month = str(mon) if int(mon) >= 10 else '0' + str(mon)
        day1 = str(day) if int(day) >= 10 else '0' + str(day)
        date = '2017-' + str(month) + '-' + str(day1)
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        if date.weekday() < 5:
            rental_rate_0 = rateDict['rental_rate_0']
            return_rate_0 = rateDict['return_rate_0']
        elif date.weekday() < 7:
            rental_rate_0 = rateDict['rental_rate_1']
            return_rate_0 = rateDict['return_rate_1']

        totalDocks = totalDocksDict[str(balanceSta)]
        t_travel = dis * 1000 / v
        t_min = math.ceil(t_travel / 60)
        t = starttime + t_min
        t_interval = t / 5
        availableBikes = station_status[str(day)][str(balanceSta)]['availableBikes']
        availableDocks = station_status[str(day)][str(balanceSta)]['availableDocks']
        rentalLost = 0
        returnLost = 0
        for i in np.arange(0, int(t_interval)):  # real-time bikes docks
            deltaNum = 0
            deltaNum = rental_rate_0[i] - return_rate_0[i]
            if float(availableBikes) < 1.0 and deltaNum > 0:
                rentalLost += deltaNum
                pass  # rental_lost += deltNum
            if float(availableDocks) < 1.0 and deltaNum < 0:
                returnLost += abs(deltaNum)
                pass  # return_lost += deltNum

            if deltaNum > 0:
                availableBikes = float(availableBikes) - deltaNum
                if availableBikes < 0:
                    availableBikes = 0
                availableDocks = float(availableDocks) + deltaNum
                if availableDocks > float(totalDocks):
                    availableBikes = 0
                    availableDocks = float(totalDocks)
            else:
                availableDocks = float(availableDocks) - abs(deltaNum)
                if availableDocks < 0:
                    availableDocks = 0
                availableBikes = float(availableBikes) + abs(deltaNum)
                if availableBikes > float(totalDocks):
                    availableDocks = 0
                    availableBikes = float(totalDocks)
        realBikes = availableBikes
        realDocks = availableDocks
        beforeBalancedTotalLost.value = beforeBalancedTotalLost.value + returnLost + rentalLost
        noise = abs(np.random.normal(loc=0.0, scale=2))
        if balanceSta in dropStation:
            if float(realBikes) >= 12:
                endtime = t + noise
                dropNum = 0
                info['realbikes'] = realBikes
            else:
                dropNum = 12 - int(realBikes)
                endtime = t + dropNum * 0.3 + noise
                info['realbikes'] = 12
        if balanceSta in pickStation:
            if float(realDocks) >= 12:
                endtime = t + noise
                pickNum = 0
                info['realbikes'] = float(totalDocks) - float(realDocks)
            else:
                pickNum = 12 - int(realDocks)
                endtime = t + pickNum * 0.3 + noise
                info['realbikes'] = float(totalDocks) - 12
        info['time'] = endtime
        realtimeBikes[str(balanceSta)] = info
        staLost = {}
        starttime = endtime
        print('drop:' + str(dropNum))
        print('pick:' + str(pickNum))
        print('distance:' + str(minDis))
        print('starttime:' + str(starttime))
        print(realtimeBikes)

        balanceStas = []
        pickStation = []
        dropStation = []
        for sta in availStations:

            t_interval = starttime / 5
            iniBikes = station_status[str(day)][str(sta)]['availableBikes']
            iniDocks = station_status[str(day)][str(sta)]['availableDocks']
            availableBikes = iniBikes
            availableDocks = iniDocks
            rateDict = rateData[str(sta)]
            month = str(mon) if int(mon) >= 10 else '0' + str(mon)
            day1 = str(day) if int(day) >= 10 else '0' + str(day)
            date = '2017-' + str(month) + '-' + str(day1)
            date = datetime.datetime.strptime(date, '%Y-%m-%d')
            if date.weekday() < 5:
                rental_rate_0 = rateDict['rental_rate_0']
                return_rate_0 = rateDict['return_rate_0']
            elif date.weekday() < 7:
                rental_rate_0 = rateDict['rental_rate_1']
                return_rate_0 = rateDict['return_rate_1']

            for i in np.arange(0, int(t_interval)):  # real-time bikes docks
                deltaNum = rental_rate_0[i] - return_rate_0[i]

                if deltaNum > 0:
                    availableBikes = float(availableBikes) - deltaNum
                    if availableBikes < 0:
                        availableBikes = 0
                    availableDocks = float(availableDocks) + deltaNum
                    if availableDocks > float(totalDocks):
                        availableBikes = 0
                        availableDocks = float(totalDocks)
                else:
                    availableDocks = float(availableDocks) - abs(deltaNum)
                    if availableDocks < 0:
                        availableDocks = 0
                    availableBikes = float(availableBikes) + abs(deltaNum)
                    if availableBikes > float(totalDocks):
                        availableDocks = 0
                        availableBikes = float(totalDocks)

            realBikes = availableBikes
            realDocks = availableDocks

            if float(realBikes) < 5:
                dropStation.append(str(sta))
                balanceStas.append(str(sta))
            if float(realDocks) < 5:
                pickStation.append(str(sta))
                balanceStas.append(str(sta))

    # getNearestNeighborLost(realtimeBikes,rateData,totalDocksDict,lostNums,station_status)
    # print(dropStation)
    # print(pickStation)
    # print(diss)


def getNearestNeighborLost(realtimeBikes, day):
    rateData = getRateData()
    station_status, totalDocksDict = getStation_status()
    # mon,day,hour = getMonthDayAndHour()
    mon = 8
    hour = 7
    position, stations_id = getPositionAndStations_id()
    balancedSta = []
    totalLost = 0
    lostNums = {}
    for sta, values in realtimeBikes.items():
        balancedSta.append(sta)
        rentalLost = 0
        returnLost = 0
        time = values['time']
        realbikes = values['realbikes']
        time_interval = time / 5

        rateDict = rateData[str(sta)]
        month = str(mon) if int(mon) >= 10 else '0' + str(mon)
        day1 = str(day) if int(day) >= 10 else '0' + str(day)
        date = '2017-' + str(month) + '-' + str(day1)
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        if date.weekday() < 5:
            rental_rate_0 = rateDict['rental_rate_0']
            return_rate_0 = rateDict['return_rate_0']
        elif date.weekday() < 7:
            rental_rate_0 = rateDict['rental_rate_1']
            return_rate_0 = rateDict['return_rate_1']

        totalDocks = int(totalDocksDict[str(sta)])
        availableBikes = realbikes
        availableDocks = float(totalDocks) - float(realbikes)
        for i in np.arange(int(time_interval), 48):  # real-time bikes docks
            deltaNum = 0
            deltaNum = rental_rate_0[i] - return_rate_0[i]

            if deltaNum > 0 and (deltaNum - float(availableBikes)) > 0:
                rentalLost += (deltaNum - float(availableBikes))
                pass  # rental_lost += deltNum
            if deltaNum < 0 and (abs(deltaNum) - float(availableDocks)) > 0:
                returnLost += (abs(deltaNum) - float(availableDocks))
                pass  # return_lost += deltNum

            if deltaNum > 0:
                availableBikes = float(availableBikes) - deltaNum
                if availableBikes < 0:
                    availableBikes = 0
                availableDocks = float(availableDocks) + deltaNum
                if availableDocks > float(totalDocks):
                    availableBikes = 0
                    availableDocks = float(totalDocks)
            else:
                availableDocks = float(availableDocks) - abs(deltaNum)
                if availableDocks < 0:
                    availableDocks = 0
                availableBikes = float(availableBikes) + abs(deltaNum)
                if availableBikes > float(totalDocks):
                    availableDocks = 0
                    availableBikes = float(totalDocks)
        lostNums[str(sta)] = rentalLost + returnLost
        totalLost += lostNums[str(sta)]

    leftStations = [sta for sta in stations_id if sta not in balancedSta]

    for sta in leftStations:
        rateDict = rateData[str(sta)]
        month = str(mon) if int(mon) >= 10 else '0' + str(mon)
        day1 = str(day) if int(day) >= 10 else '0' + str(day)
        date = '2017-' + str(month) + '-' + str(day1)
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        if date.weekday() < 5:
            rental_rate_0 = rateDict['rental_rate_0']
            return_rate_0 = rateDict['return_rate_0']
        elif date.weekday() < 7:
            rental_rate_0 = rateDict['rental_rate_1']
            return_rate_0 = rateDict['return_rate_1']

        iniBikes = station_status[str(day)][str(sta)]['availableBikes']
        iniDocks = station_status[str(day)][str(sta)]['availableDocks']
        totalDocks = totalDocksDict[str(sta)]
        availableBikes = iniBikes
        availableDocks = iniDocks
        rentalLost = 0
        returnLost = 0
        for i in np.arange(0, 48):  # real-time bikes docks
            deltaNum = 0
            deltaNum = rental_rate_0[i] - return_rate_0[i]

            if deltaNum > 0 and (deltaNum - float(availableBikes)) > 0:
                rentalLost += (deltaNum - float(availableBikes))
                pass  # rental_lost += deltNum
            if deltaNum < 0 and (abs(deltaNum) - float(availableDocks)) > 0:
                returnLost += (abs(deltaNum) - float(availableDocks))
                pass  # return_lost += deltNum

            if deltaNum > 0:
                availableBikes = float(availableBikes) - deltaNum
                if availableBikes < 0:
                    availableBikes = 0
                availableDocks = float(availableDocks) + deltaNum
                if availableDocks > float(totalDocks):
                    availableBikes = 0
                    availableDocks = float(totalDocks)
            else:
                availableDocks = float(availableDocks) - abs(deltaNum)
                if availableDocks < 0:
                    availableDocks = 0
                availableBikes = float(availableBikes) + abs(deltaNum)
                if availableBikes > float(totalDocks):
                    availableDocks = 0
                    availableBikes = float(totalDocks)
        lostNums[str(sta)] = rentalLost + returnLost
        totalLost += lostNums[str(sta)]
    lostNums['totalLost'] = totalLost
    print(totalLost)
    return lostNums


def nearestNeihborReposition():
    experiment_path = './bike_sharing_data/mydata/nearestNeihborReposition'
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    # month, day, hour = getMonthDayAndHour()
    mon = 8

    hour = 7
    for day in range(1, 32):
        realtimeBikes = multiprocessing.Manager().dict()
        position, stations_id = getPositionAndStations_id()
        availStations = multiprocessing.Manager().list(stations_id)
        beforeBalancedTotalLost = multiprocessing.Value("d", 0)
        startStation1 = '237'
        startStation2 = '369'
        lostNums = {}
        mutex = multiprocessing.Lock()
        p1 = multiprocessing.Process(target=nearestNeihborRepositionStart, args=(
            startStation1, availStations, mutex, realtimeBikes, day, beforeBalancedTotalLost))
        p2 = multiprocessing.Process(target=nearestNeihborRepositionStart, args=(
            startStation2, availStations, mutex, realtimeBikes, day, beforeBalancedTotalLost))

        p1.start()
        p2.start()

        p1.join(9)
        p2.join(9)
        print(realtimeBikes)
        lostNums = getNearestNeighborLost(realtimeBikes, day)
        lostNums['totalLost'] += beforeBalancedTotalLost.value
        print(lostNums)
        filename = 'nearestNeihborRepositionLost_month_' + str(mon) + '_day_' + str(day) + '_hour_' + str(
            78910) + '.json'
        with open(os.path.join(experiment_path, filename), 'w') as f:
            json.dump(lostNums, f)
        print('day' + str(day) + 'finished!')


def nearestNeihborBaseServiceLevelRepositionStart(startStation, availStations, mutex, realtimeBikes,
                                                  visitedPath, visitedDis, balanceNum, beforeBalancedTotalLost, day):
    position, stations_id = getPositionAndStations_id()
    rateData = getRateData()
    station_status, totalDocksDict = getStation_status()
    # mon, day, hour = getMonthDayAndHour()
    mon = 8

    hour = 7
    dropStation = []
    pickStation = []
    balanceStas = []

    for sta in stations_id:
        iniBikes = station_status[str(day)][str(sta)]['availableBikes']
        iniDocks = station_status[str(day)][str(sta)]['availableDocks']
        totalDocks = totalDocksDict[str(sta)]
        serviceLevel = []
        rateDict = rateData[str(sta)]
        month = str(mon) if int(mon) >= 10 else '0' + str(mon)
        day1 = str(day) if int(day) >= 10 else '0' + str(day)
        date = '2017-' + str(month) + '-' + str(day1)
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        if date.weekday() < 5:
            rental_rate_0 = rateDict['rental_rate_0']
            return_rate_0 = rateDict['return_rate_0']
        elif date.weekday() < 7:
            rental_rate_0 = rateDict['rental_rate_1']
            return_rate_0 = rateDict['return_rate_1']

        for docks in range(1, int(totalDocks)):
            availableBikes = int(totalDocks) - docks
            availableDocks = docks
            flag = 0
            for j in np.arange(0, 19):
                deltaNum = 0
                if j >= 48:
                    break
                else:
                    deltaNum = rental_rate_0[j] - return_rate_0[j]
                if deltaNum > 0:
                    availableBikes = float(availableBikes) - deltaNum
                    if availableBikes <= 1:
                        flag = 1
                        break
                    availableDocks = float(availableDocks) + deltaNum
                    if availableDocks >= float(totalDocks) - 1:
                        flag = 1
                        break
                else:
                    availableDocks = float(availableDocks) - abs(deltaNum)
                    if availableDocks <= 1:
                        flag = 1
                        break
                    availableBikes = float(availableBikes) + abs(deltaNum)
                    if availableBikes >= float(totalDocks) - 1:
                        flag = 1
                        break
            if flag == 0:
                serviceLevel.append(int(totalDocks) - int(docks))
        if not serviceLevel:
            pickStation.append(str(sta))
            balanceStas.append(str(sta))
        else:
            if float(iniBikes) < min(serviceLevel):
                dropStation.append(str(sta))
                balanceStas.append(str(sta))
            if float(iniDocks) > max(serviceLevel):
                pickStation.append(str(sta))
                balanceStas.append(str(sta))
    # balanceSta = startStation
    visitedPath.append(startStation)
    starttime = 0
    v = 7
    while True:

        info = {}
        minDis = 10
        pickNum = 0
        dropNum = 0
        print('balanceStas' + str(balanceStas))
        if not balanceStas:
            break
        mutex.acquire()
        balanceStas = [s for s in balanceStas if s in availStations]
        if not balanceStas:
            break
        for sta in balanceStas:
            dis = manhattan_distance(position[str(startStation)][0], position[str(startStation)][1], position[sta][0],
                                     position[sta][1])
            if dis < minDis:
                minDis = dis
                balanceSta = sta
        startStation = balanceSta
        availStations.remove(str(balanceSta))
        mutex.release()

        rateDict = rateData[str(balanceSta)]
        month = str(mon) if int(mon) >= 10 else '0' + str(mon)
        day1 = str(day) if int(day) >= 10 else '0' + str(day)
        date = '2017-' + str(month) + '-' + str(day1)
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        if date.weekday() < 5:
            rental_rate_0 = rateDict['rental_rate_0']
            return_rate_0 = rateDict['return_rate_0']
        elif date.weekday() < 7:
            rental_rate_0 = rateDict['rental_rate_1']
            return_rate_0 = rateDict['return_rate_1']

        totalDocks = totalDocksDict[str(balanceSta)]
        t_travel = dis * 1000 / v
        t_min = math.ceil(t_travel / 60)
        t = starttime + t_min
        t_interval = t / 5
        availableBikes = station_status[str(day)][str(balanceSta)]['availableBikes']
        availableDocks = station_status[str(day)][str(balanceSta)]['availableDocks']
        rentalLost = 0
        returnLost = 0
        for i in np.arange(0, int(t_interval)):  # real-time bikes docks
            deltaNum = rental_rate_0[i] - return_rate_0[i]

            if float(availableBikes) < 1.0 and deltaNum > 0:
                rentalLost += deltaNum
                pass  # rental_lost += deltNum
            if float(availableDocks) < 1.0 and deltaNum < 0:
                returnLost += abs(deltaNum)
                pass  # return_lost += deltNum

            if deltaNum > 0:
                availableBikes = float(availableBikes) - deltaNum
                if availableBikes < 0:
                    availableBikes = 0
                availableDocks = float(availableDocks) + deltaNum
                if availableDocks > float(totalDocks):
                    availableBikes = 0
                    availableDocks = float(totalDocks)
            else:
                availableDocks = float(availableDocks) - abs(deltaNum)
                if availableDocks < 0:
                    availableDocks = 0
                availableBikes = float(availableBikes) + abs(deltaNum)
                if availableBikes > float(totalDocks):
                    availableDocks = 0
                    availableBikes = float(totalDocks)
        mutex.acquire()
        beforeBalancedTotalLost.value = beforeBalancedTotalLost.value + rentalLost + returnLost
        mutex.release()
        realBikes = availableBikes
        realDocks = availableDocks

        totalDocks = totalDocksDict[str(balanceSta)]
        serviceLevel = []
        for docks in range(1, int(totalDocks)):
            availableBikes = int(totalDocks) - docks
            availableDocks = docks
            flag = 0
            for j in np.arange(0, 19):
                if j >= 48:
                    break
                else:
                    deltaNum = rental_rate_0[j] - return_rate_0[j]
                if deltaNum > 0:
                    availableBikes = float(availableBikes) - deltaNum
                    if availableBikes <= 1:
                        flag = 1
                        break
                    availableDocks = float(availableDocks) + deltaNum
                    if availableDocks >= float(totalDocks) - 1:
                        flag = 1
                        break
                else:
                    availableDocks = float(availableDocks) - abs(deltaNum)
                    if availableDocks <= 1:
                        flag = 1
                        break
                    availableBikes = float(availableBikes) + abs(deltaNum)
                    if availableBikes >= float(totalDocks) - 1:
                        # print('availableBikes:'+str(availableBikes))
                        flag = 1
                        break
            if flag == 0:
                serviceLevel.append(int(totalDocks) - int(docks))
        noise = abs(np.random.normal(loc=0.0, scale=2))
        if balanceSta in dropStation:
            if min(serviceLevel) <= float(realBikes) <= max(serviceLevel):
                endtime = t + noise
                dropNum = 0
                info['realbikes'] = realBikes
            else:
                dropNum = min(serviceLevel) - math.floor(float(realBikes))
                endtime = t + dropNum * 0.3 + noise
                info['realbikes'] = min(serviceLevel)
        if balanceSta in pickStation:
            if float(realBikes) <= max(serviceLevel):
                endtime = t + noise
                pickNum = 0
                info['realbikes'] = float(realBikes)
            else:
                pickNum = math.floor(float(realBikes)) - max(serviceLevel)
                endtime = t + pickNum * 0.3 + noise
                info['realbikes'] = max(serviceLevel)
        info['time'] = endtime
        realtimeBikes[str(balanceSta)] = info
        starttime = endtime
        print('balanceSta:' + str(balanceSta))
        print('drop:' + str(dropNum))
        print('pick:' + str(pickNum))
        print('distance:' + str(minDis))
        print('starttime:' + str(starttime))
        print(realtimeBikes)
        visitedPath.append(balanceSta)
        visitedDis.append(minDis)
        if int(dropNum) > 0:
            balanceNum[str(balanceSta)] = -dropNum
        elif int(pickNum) > 0:
            balanceNum[str(balanceSta)] = pickNum
        else:
            balanceNum[str(balanceSta)] = 0

        balanceStas = []
        pickStation = []
        dropStation = []
        for sta in availStations:

            t_interval = starttime / 5
            iniBikes = station_status[str(day)][str(sta)]['availableBikes']
            iniDocks = station_status[str(day)][str(sta)]['availableDocks']
            availableBikes = iniBikes
            availableDocks = iniDocks
            totalDocks = totalDocksDict[str(sta)]

            rateDict = rateData[str(sta)]
            month = str(mon) if int(mon) >= 10 else '0' + str(mon)
            day1 = str(day) if int(day) >= 10 else '0' + str(day)
            date = '2017-' + str(month) + '-' + str(day1)
            date = datetime.datetime.strptime(date, '%Y-%m-%d')
            if date.weekday() < 5:
                rental_rate_0 = rateDict['rental_rate_0']
                return_rate_0 = rateDict['return_rate_0']
            elif date.weekday() < 7:
                rental_rate_0 = rateDict['rental_rate_1']
                return_rate_0 = rateDict['return_rate_1']

            for i in np.arange(0, int(t_interval)):  # real-time bikes docks
                deltaNum = rental_rate_0[i] - return_rate_0[i]

                if deltaNum > 0:
                    availableBikes = float(availableBikes) - deltaNum
                    if availableBikes < 0:
                        availableBikes = 0
                    availableDocks = float(availableDocks) + deltaNum
                    if availableDocks > float(totalDocks):
                        availableBikes = 0
                        availableDocks = float(totalDocks)
                else:
                    availableDocks = float(availableDocks) - abs(deltaNum)
                    if availableDocks < 0:
                        availableDocks = 0
                    availableBikes = float(availableBikes) + abs(deltaNum)
                    if availableBikes > float(totalDocks):
                        availableDocks = 0
                        availableBikes = float(totalDocks)

            realBikes = availableBikes
            realDocks = availableDocks

            serviceLevel = []
            for docks in range(1, int(totalDocks)):
                availableBikes = int(totalDocks) - docks
                availableDocks = docks
                flag = 0
                for j in np.arange(0, 19):
                    deltaNum = 0
                    if j >= 48:
                        break
                    else:
                        deltaNum = rental_rate_0[j] - return_rate_0[j]
                    if deltaNum > 0:
                        availableBikes = float(availableBikes) - deltaNum
                        if availableBikes <= 1:
                            flag = 1
                            break
                        availableDocks = float(availableDocks) + deltaNum
                        if availableDocks >= float(totalDocks) - 1:
                            flag = 1
                            break
                    else:
                        availableDocks = float(availableDocks) - abs(deltaNum)
                        if availableDocks <= 1:
                            flag = 1
                            break
                        availableBikes = float(availableBikes) + abs(deltaNum)
                        if availableBikes >= float(totalDocks) - 1:
                            # print('availableBikes:'+str(availableBikes))
                            flag = 1
                            break
                if flag == 0:
                    serviceLevel.append(int(totalDocks) - int(docks))

            if not serviceLevel:
                pickStation.append(str(sta))
                balanceStas.append(str(sta))
            else:
                if float(realBikes) < min(serviceLevel):
                    dropStation.append(str(sta))
                    balanceStas.append(str(sta))
                if float(realBikes) > max(serviceLevel):
                    pickStation.append(str(sta))
                    balanceStas.append(str(sta))
        print('dropStation:' + str(dropStation))
        print('pickStation:' + str(pickStation))


def nearestNeihborBaseServiceLevelReposition():
    experiment_path = './bike_sharing_data/mydata/nearestNeihborRepositionBasedService'
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    # month, day, hour = getMonthDayAndHour()
    mon = 8
    hour = 7
    for day in range(1, 32):
        realtimeBikes = multiprocessing.Manager().dict()
        visitedPath1 = multiprocessing.Manager().list()
        visitedDis1 = multiprocessing.Manager().list()
        balanceNum1 = multiprocessing.Manager().dict()
        visitedPath2 = multiprocessing.Manager().list()
        visitedDis2 = multiprocessing.Manager().list()
        balanceNum2 = multiprocessing.Manager().dict()
        position, stations_id = getPositionAndStations_id()
        availStations = multiprocessing.Manager().list(stations_id)
        beforeBalancedTotalLost = multiprocessing.Value("d", 0)
        startStation1 = '237'
        startStation2 = '369'
        lostNums = {}
        mutex = multiprocessing.Lock()
        p1 = multiprocessing.Process(target=nearestNeihborBaseServiceLevelRepositionStart,
                                     args=(
                                         startStation1, availStations, mutex, realtimeBikes, visitedPath1, visitedDis1,
                                         balanceNum1, beforeBalancedTotalLost, day))
        p2 = multiprocessing.Process(target=nearestNeihborBaseServiceLevelRepositionStart,
                                     args=(
                                         startStation2, availStations, mutex, realtimeBikes, visitedPath2, visitedDis2,
                                         balanceNum2, beforeBalancedTotalLost, day))

        p1.start()
        p2.start()

        p1.join(8)
        p2.join(8)
        print(realtimeBikes)
        lostNums = getNearestNeighborLost(realtimeBikes, day)

        experimentResult = {}
        truck1 = {}
        truck2 = {}
        truck1['visitedPath'] = list(visitedPath1)
        truck1['visitedDis'] = list(visitedDis1)
        balanceNum11 = {}
        for sta, values in balanceNum1.items():
            balanceNum11[str(sta)] = values
        truck1['balanceNum'] = balanceNum11

        truck2['visitedPath'] = list(visitedPath2)
        truck2['visitedDis'] = list(visitedDis2)
        balanceNum22 = {}
        for sta, values in balanceNum2.items():
            balanceNum22[str(sta)] = values
        truck2['balanceNum'] = balanceNum22

        experimentResult['truck1'] = truck1
        experimentResult['truck2'] = truck2
        lostNums['totalLost'] += beforeBalancedTotalLost.value
        experimentResult['lost'] = lostNums

        print('through stations:' + str(visitedPath1))
        print('balanced number:' + str(balanceNum1))
        print('travel distance:' + str(visitedDis1))

        print('through stations:' + str(visitedPath2))
        print('balanced number:' + str(balanceNum2))
        print('travel distance:' + str(visitedDis2))

        print("beforeBalancedTotalLost:" + str(beforeBalancedTotalLost.value))
        print('total customer loss:' + str(lostNums['totalLost']))
        print('lostNums:' + str(lostNums))
        filename = 'nearestNeihborRepositionResult_month_' + str(mon) + '_day_' + str(day) + '_hour_' + str(
            78910) + '.json'
        with open(os.path.join(experiment_path, filename), 'w') as f:
            json.dump(experimentResult, f)
        print('day' + str(day) + 'finished!')


if __name__ == '__main__':
    mctsAlgorithm()
    # noReposition()
    # staticReposition()
    # nearestNeihborReposition()
    # nearestNeihborBaseServiceLevelReposition()
