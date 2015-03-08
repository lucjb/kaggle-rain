import csv
import numpy as np
import matplotlib.pyplot as plt
import math

thresholds = np.arange(70)
def heaviside(actual):
    return thresholds >= actual

def calc_crps(predictions, actuals):
    obscdf = np.array([heaviside(i) for i in actuals])
    crps = np.mean(np.mean((predictions - obscdf) ** 2))
    return crps

def sigmoid(center, length):
    xs = np.arange(length)
    return 1. / (1 + np.exp(-(xs - center)))

def cdfs(means):
	cdfs = []
	for estimated_mean_rr in means:
		if estimated_mean_rr <= 0:
			cdfs.append([1]*70)	
		else:
			cdfs.append(sigmoid(estimated_mean_rr, 70))	
	return cdfs

def parse_floats(row, col_ind):
	return np.array(row[col_ind].split(' '), dtype='float')

def parse_rr(row, rr_ind):
	return parse_floats(row, rr_ind)

def split_radars(times):
	T = []
	j=1
	s=0
	while j<len(times):
		if times[j]>=times[j-1]:
			T.append(range(s,j))
			s = j 
		j+=1
	T.append(range(s,j))
	return T

def closest_slice(distances):	
	min_distance = np.min(distances)
	closest_slice = []
	for j, d in enumerate(distances):
		if d == min_distance:
			closest_slice.append(j)	
	return closest_slice, min_distance
		
def time_p(radar_ind, times, rr):
	max_t = -1.
	min_t =	100.
		
	for ri in radar_ind:
		if rr[ri]>0 and times[ri]<min_t:
			min_t=times[ri]
		if rr[ri]>0 and times[ri]>max_t:
			max_t=times[ri]
	if max_t<0:
		return 0
	b = (times.max()-times.min())/2/len(times)
	time_period = (max_t-min_t + b)/60
	return time_period

def mean_without_zeros(a):
	filtered = a[a!=0]
	if len(filtered)==0:
		return 0
	return filtered.mean()


def closest_good_estimate_w(rr, distances, radar_indices, w):
	ws = []
	for we in w:
		if we<=1:
			ws.append(we)
		else:
			ws.append(0)
	w = np.array(ws)
	r, _ = closest_slice(distances)
	q = np.sum(w[r])					
	if q>0:
		v = np.average(rr[r], weights=w[r])
	else:
		v=-1
	min_dis = 100000
	for radar in radar_indices:
		q = np.sum(w[radar])					
		if q>0:
			v2 = np.average(rr[radar], weights=w[radar])
		else:
			v2=-1
		dis = distances[radar][0]					
		if v2>=0 and dis<min_dis:
			r=radar
			min_dis=dis
			v = v2
	return r, v

def closest_good_estimate(rr, distances, radar_indices, w):
	r = radar_indices[0]
	v = -1
	min_dis = 100000
	for radar in radar_indices:
		v2 = np.average(rr[radar])
		dis = distances[radar][0]
		q = np.sum(w[radar])					
		if v2>=0 and dis<min_dis and q>0 and q<999*(len(radar)-1):
			r=radar
			min_dis=dis
			v = v2
	return r, v

def clean_weights(w, filler=0):
	clean = []
	for x in w:
		if x>=0 and x<=1:
			clean.append(x)
		else:
			clean.append(filler)
	return clean

def all_good_estimates(rr, distances, radar_indices, w, times):
	good = []
	dsum = np.sum(distances)
	if dsum==0: dsum=1
	for radar in radar_indices:
		v2 = np.average(rr[radar])
		dis = distances[radar][0]
		q = np.sum(w[radar])					
		if v2>=0 and q>1:
			time_period = time_p(radar, times, rr)
			e = v2*time_period
			good.append(e)
	return good


def data_set(file_name):
    reader = csv.reader(open(file_name))

    header = reader.next()
    id_ind = header.index('Id')
    rr1_ind = header.index('RR1')
    rr2_ind = header.index('RR2')
    rr3_ind = header.index('RR3')
    time_ind = header.index('TimeToEnd')
    rad_q_ind = header.index('RadarQualityIndex')	
    try:
    	expected_ind = header.index('Expected')
    except ValueError:
	# no label
	expected_ind = -1
	
    composite_ind = header.index('Composite')
    distance_ind = header.index('DistanceToRadar')
    X = []
    y = []
    ids = []
    avgs = []

    for i, row in enumerate(reader):
	ids.append(row[id_ind])	        
	times = parse_floats(row, time_ind)
	distances = parse_floats(row, time_ind)
        rr1 = parse_rr(row, rr1_ind)
        rr2 = parse_rr(row, rr2_ind)
        rr3 = parse_rr(row, rr3_ind)
	w = parse_floats(row, rad_q_ind)
	
	if expected_ind >= 0:
		ey = float(row[expected_ind])
		y.append(ey)

	
	radar_indices = split_radars(times)
	
	good = []
	good.extend(all_good_estimates(rr1, distances, radar_indices, w, times))
	good.extend(all_good_estimates(rr2, distances, radar_indices, w, times))	
	good.extend(all_good_estimates(rr3, distances, radar_indices, w, times))	

	if len(good)==0:	
		avgs.append(0.)
	else:
		avg = np.mean(good)
		avgs.append(avg)
	
        if i % 10000 == 0:
            print "Completed row %d" % i
    return ids, np.array(y), np.array(avgs)

#0.00992382187229 -> 0.00971819
#0.00983595164706 -> 0.00962434
#0.00957061504447 -> 0.00924509
#0.00952959922595 -> 0.00918081
#0.0095278045182
#0.00945252983071
#0.0094347918118 -> 0.00900467
#0.00941938258085 -> 0.00893021
#0.00938841086168 
#0.00923516814223
#0.00923510027563
#0.00922980704588 -> 0.00871121
#0.00922233467044
#Baseline CRPS: 0.00965034244803
#1126695 training examples
#987398 0s
#133717 valid no 0
#5580 invalid

_, y, avgs = data_set('train_2013.csv')
print 'CRPS: ',  calc_crps(cdfs(avgs), y)
print 'RMSE', math.sqrt(np.mean((y[y<100]-avgs[y<100])**2))

plt.scatter(avgs[y<100], y[y<100])
#plt.hist(y[y<100], log=True)
plt.show()


print 'Predicting for sumbission...'
print 'Loading test file...'
ids, _, avgs = data_set('test_2014.csv')
cdfs = cdfs(avgs)

print 'Writing submision file...'

writer = csv.writer(open('unsupervised-sub.csv', 'w'))
solution_header = ['Id']
solution_header.extend(['Predicted{0}'.format(t) for t in xrange(0, 70)])

writer.writerow(solution_header)

for i, id in enumerate(ids):
	prediction = cdfs[i]	
        solution_row = [id]
        solution_row.extend(prediction)
        writer.writerow(solution_row)

        if i % 10000 == 0:
            print "Completed row %d" % i	

