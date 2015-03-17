import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats
from math import *
from scipy import interpolate
import scipy.signal
from scipy.integrate import simps

thresholds = np.arange(70)
def heaviside(actual):
    return thresholds >= actual

def erfcc(x):
    """Complementary error function."""
    z = abs(x)
    t = 1. / (1. + 0.5*z)
    r = t * exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196+
    	t*(.09678418+t*(-.18628806+t*(.27886807+
    	t*(-1.13520398+t*(1.48851587+t*(-.82215223+
    	t*.17087277)))))))))
    if (x >= 0.):
    	return r
    else:
    	return 2. - r

def normcdf(x, mu, sigma):
    t = x-mu;
    y = 0.5*erfcc(-t/(sigma*sqrt(2.0)));
    if y>1.0:
        y = 1.0;
    return y

def gauss(mean, l, v=1):
	xs = np.arange(l)
	return [normcdf(x, mean, v) for x in xs]
	
def calc_crps(predictions, actuals):
    obscdf = np.array([heaviside(i) for i in actuals])
    crps = np.mean(np.mean((predictions - obscdf) ** 2))
    return crps

def step(center, length=70):
	x = [1.]*length
	for i in range(0, int(center)+1):
		x[i]=0.
	return np.array(x)

def sigmoid(center, length):
    xs = np.arange(length)
    return 1. / (1 + np.exp(-(xs - center)))

def cdfs(means):
	cdfs = []
	for estimated_mean_rr in means:
		if estimated_mean_rr <= 0:
			cdfs.append([1]*70)
		elif estimated_mean_rr>70:
			a = [0]*69
			a.append(1)
			cdfs.append(a)		
		else:
			s = gauss(estimated_mean_rr, 70)
			cdfs.append(s)	
	return cdfs

def parse_floats(row, col_ind):
	return np.array(row[col_ind].split(' '), dtype='float')


def parse_rr(row, rr_ind, default=None):
	if default:
	 	a  = parse_floats(row, rr_ind)
		for i, v in enumerate(a):
			if v<0 or v>1000:
				a[i]=default
		return a
	else:
		return parse_floats(row, rr_ind)


def split_radars(times):
	T = []
	j=1
	s=0
	while j<len(times):
		if times[j]!=times[j-1]:
			T.append(range(s,j))
			s = j 
		j+=1
	T.append(range(s,j))
	return T
		
def mean_without_zeros(a):
	filtered = a[a!=0]
	if len(filtered)==0:
		return 0
	return filtered.mean()


def clean_radar_q(w, filler=0):
	clean = []
	for x in w:
		if x>=0 and x<=1:
			clean.append(x)
		else:
			clean.append(filler)
	return w

def hmdir_(times, rr, w, x):
	valid_t = times[rr>=0]
	q = [0.5]*len(valid_t)
	for ai, a in enumerate(w[rr>=0]):
		if a<=1:
			q[ai]=a
	valid_r = rr[rr>=0]*q
		
	if len(valid_t)<2: return valid_r[0]/60.
	f = interpolate.interp1d(valid_t, valid_r)
	ra = range(int(valid_t.min()), int(valid_t.max()+1))	
	tl = f(ra)
	#if len(tl)>=9:
	#	tl = scipy.signal.savgol_filter(tl, min(len(tl), 9), 3)
	est = sum(tl)/60.
	return est
	
	
def hmdir(times, rr, w, x):
	hour = [0.]*61
	for i in range(1, len(times)):
		for j in range(int(times[len(times)-i]), int(times[len(times)-i-1])):
			v = rr[len(times)-i-1]
			q = w[len(times)-i-1]
			xi = x[len(times)-i-1] 
			if q !=1: q = 0.5
			if v>=0 and v<100 and not xi in [6, 8]:
				hour[j]=v*q

	est = sum(hour)/60.
	return est 

def all_good_estimates(rr, distances, radar_indices, w, times, hts):
	asd = []
	ds = []
	dt = float(sum(distances))
	for radar in radar_indices:
		rain = rr[radar]
		rr_error_rate = len(rain[rain<0])/float(len(rain))
		if rr_error_rate<0.5:
			d = distances[radar][0]
			est = hmdir(times[radar], rr[radar], w[radar], hts[radar])
			asd.append(est)
			ds.append(d)
	return asd, ds


def mean(x, default=0):
	if len(x)==0: return default
	return np.mean(x)


def is_cdf_valid(case):
    if case[0] < 0 or case[0] > 1:
        return False
    for i in xrange(1, len(case)):
        if case[i] > 1 or case[i] < case[i-1]:
            return False
    return True

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
    hydro_type_ind = header.index('HydrometeorType')
    y = []
    ids = []
    avgs = []

    for i, row in enumerate(reader):
	ids.append(row[id_ind])	        
	times = parse_floats(row, time_ind)
	distances = parse_floats(row, distance_ind)
        rr1 = parse_rr(row, rr1_ind)
        rr2 = parse_rr(row, rr2_ind)
        rr3 = parse_rr(row, rr3_ind)
	w = parse_floats(row, rad_q_ind)
	hidro_types = parse_floats(row, hydro_type_ind)	

	if expected_ind >= 0:
		ey = float(row[expected_ind])
		y.append(ey)

	
	radar_indices = split_radars(distances)
	
	good = []
	good_d = []
	rr1_estimates, d1 = all_good_estimates(rr1, distances, radar_indices, w, times, hidro_types)
	rr2_estimates, d2 = all_good_estimates(rr2, distances, radar_indices, w, times, hidro_types)
	rr3_estimates, d3 = all_good_estimates(rr3, distances, radar_indices, w, times, hidro_types)
	good.extend(rr1_estimates)
	good.extend(rr2_estimates)
	good.extend(rr3_estimates)
	good_d.extend(d1)	
	good_d.extend(d2)
	good_d.extend(d3)
	if len(good)>0:
		if np.mean(good)==0:
			s = [1]*70
			avgs.append(s)
		else:
			h = []	
			for j, x in enumerate(good):	
				s = sigmoid(round(x), 70)
				h.append(s)
			h = np.reshape(h, (len(good), 70))
			
			total = np.average(h, axis=0)
			if not is_cdf_valid(total):
				plt.plot(total)
				plt.show()			
			avgs.append(total)
	else:
		s = [1]*70
		avgs.append(s)

	if i % 10000 == 0:
		print "Completed row %d" % i
    return ids, np.array(y), np.array(avgs)

#0.00904234862754
#0.00904150831178
#0.00904983861228
#0.00901613263585
#0.00900412724833
#0.00900165157547
#0.00900155415651
#0.00900142821889
#0.00899566077666
#0.00899140245563
#0.00899136162509
#0.00899121498571
#0.00898516450647
#0.00898631930177
#0.00898616252983

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
#0.0092045862579
#0.00920457357894 -> 0.00867324
#0.0092016208212
#0.00920147312222 -> 0.00867015
#0.00920130043796
#0.00919861298415
#0.00919647579626
#0.00919475970769
#0.00919475679584

#0.00916480687816 -> 0.00861711
#0.00915106704337
#0.00913163690433 -> 0.00855668
#0.00912739404781
#0.00912342542954
#0.00912337214931
#0.00912001249288 -> 0.00853307
#0.00910320309268 -> 0.00849574


#Baseline CRPS: 0.00965034244803
#1126695 training examples
#987398 0s
#133717 valid no 0
#5580 invalid

_, y, avgs = data_set('train_2013.csv')
print 'CRPS: ',  calc_crps(avgs, y)


print 'Predicting for sumbission...'
print 'Loading test file...'
ids, _, avgs = data_set('test_2014.csv')
cdfs = avgs

print 'Writing submision file...'

writer = csv.writer(open('unsupervised-cdfagv-sub.csv', 'w'))
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

