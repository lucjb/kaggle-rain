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


def split_radars(distances, times):
	T = []
	j=1
	s=0
	while j<len(distances):
		if distances[j]!=distances[j-1] or times[j]>=times[j-1]:
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

def hmdir_(times, rr, w, x, d):
	valid_t = times[(rr>=0)&(rr<100)]
	valid_r = rr[(rr>=0)&(rr<100)]
	
	q = [0.5]*len(valid_t)
	for ai, a in enumerate(w[(rr>=0)&(rr<100)]):
		if a==1:
			q[ai]=1
	valid_r = valid_r*q
	if len(valid_t)==0: return 0	
	if len(valid_t)<2: return valid_r[0]/60.
	f = interpolate.interp1d(valid_t, valid_r)
	ra = range(int(valid_t.min()), int(valid_t.max()+1))	
	tl = f(ra)
	#plt.plot(tl)

	if len(tl)>=11:
		tl = scipy.signal.savgol_filter(tl, min(len(tl), 11), 4)
		
	#plt.plot(tl)
	#plt.show()
	est = sum(tl)/60.
	return est
	
	
def hmdir(times, rr, w, hts, distances, ey, defaults):
	hour = [0.]*61
	for i in range(1, len(times)):
		for j in range(int(times[len(times)-i]), int(times[len(times)-i-1])):
			v = rr[len(times)-i-1]
			q = w[len(times)-i-1]
			ht = hts[len(times)-i-1] 
			if q!=1: q = 0.5
			if v>=0 and v<100 and not ht in [6, 8]:
				hour[j]=v*q
			elif ht == 1:
				hour[j]=defaults[0]
			elif ht == 2:
				hour[j]=defaults[1]
			elif ht == 3:
				hour[j]=defaults[2]

			
	est = sum(hour)/60.
	return est

def all_good_estimates(rr, distances, radar_indices, w, times, hts, ey, defaults):
	age = []
	agd = []
	for radar in radar_indices:
		rain = rr[radar]
		q = w[radar]
		rr_error_rate = len(rain[(rain<0)])/float(len(rain))
		bad_q_rate = len(q[(q==0.)])/float(len(q))
		q_error_rate = len(q[(q>1.)])/float(len(q))
		if rr_error_rate<0.5:	
			est = hmdir(times[radar], rr[radar], w[radar], hts[radar], distances[radar], ey, defaults)
			age.append(est)
			agd.append(distances[radar][0])
		else:
			'''
			rt = hts[radar]
			rt_rate = len(rt[(rt>1)&(rt<4)])/float(len(rt))
			if rt_rate>0.05:
				age.append(20)
			else:
				age.append(0)
			agd.append(distances[radar][0])
			'''
	return age, agd


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

def avg_cdf(h):
	h = np.reshape(h, (len(h), 70))
	total = np.average(h, axis=0)
	return total

def estimate_cdf(good):
	cdf = None
	if len(good)>0:
		if np.mean(good)==0:
			cdf = [1]*70
		else:
			h = []	
			for j, x in enumerate(good):	
				s = sigmoid(round(x), 70)
				h.append(s)
			total = avg_cdf(h)
			cdf = total
	else:
		cdf = [1]*70
	
	return cdf


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
    errors = []
    error_distances = []
    g = 0
    rain_types1 = []
    rain_types2 = []
    rain_types3 = []

    for i, row in enumerate(reader):
	ids.append(row[id_ind])	        
	times = parse_floats(row, time_ind)
	distances = parse_floats(row, distance_ind)
        rr1 = parse_rr(row, rr1_ind)
        rr2 = parse_rr(row, rr2_ind)
        rr3 = np.fabs(parse_rr(row, rr3_ind))
	w = parse_floats(row, rad_q_ind)
	hidro_types = parse_floats(row, hydro_type_ind)	
	
	if expected_ind >= 0:
		ey = float(row[expected_ind])
		y.append(ey)
	else:
		ey = -1
	
	radar_indices = split_radars(distances, times)
	'''
	for radar in radar_indices:
		rht = hidro_types[radar]
		for j, x in enumerate(rht):
			if x==1:
				rain_types1.append(rr2[radar][j])
			elif x==2:
				rain_types2.append(rr2[radar][j])
			elif x==3:
				rain_types3.append(rr2[radar][j])
	'''
	
	good = []
	rr1_estimates, rr1_d = all_good_estimates(rr1, distances, radar_indices, w, times, hidro_types, ey, [0.33, 33.31, 33.31])
	rr2_estimates, rr2_d  = all_good_estimates(rr2, distances, radar_indices, w, times, hidro_types, ey, [1.51, 36.37, 81.17])
	rr3_estimates, rr3_d  = all_good_estimates(rr3, distances, radar_indices, w, times, hidro_types, ey, [4.52, 38.60, 42.34])
	good.extend(rr1_estimates)
	good.extend(rr2_estimates)
	good.extend(rr3_estimates)
		
	cdfs = []
	cdfs.append(estimate_cdf(rr1_estimates))
	cdfs.append(estimate_cdf(rr2_estimates))
	cdfs.append(estimate_cdf(rr3_estimates))
	avgs.append(avg_cdf(cdfs))
	'''
	if ey>=0 and ey<100:
		errors.extend(np.array(good)-ey)
		error_distances.extend(rr1_d)
		error_distances.extend(rr2_d)
		error_distances.extend(rr3_d)
     	'''
	if i % 10000 == 0:
		print "Completed row %d" % i
    '''
    errors = np.array(errors)
    error_distances = np.array(error_distances)
    plt.hist(errors[error_distances<10], bins=100)
    plt.hist(errors[(error_distances>=10)&(error_distances<20)], bins=100, color='red')
    plt.hist(errors[(error_distances>=20)&(error_distances<30)], bins=100, color='green')
    plt.hist(errors[(error_distances>=30)&(error_distances<40)], bins=100, color='black')
    print 'asdsad', g
    plt.show()

    rain_types1=np.array(rain_types1)
    print np.median(rain_types1[(rain_types1>=0)&(rain_types1<1000)])
    plt.hist(rain_types1[(rain_types1>=0)&(rain_types1<1000)])
    plt.show()

    rain_types2=np.array(rain_types2)
    print np.median(rain_types2[(rain_types2>=0)&(rain_types2<1000)])
    plt.hist(rain_types2[(rain_types2>=0)&(rain_types2<1000)])
    plt.show()


    rain_types3=np.array(rain_types3)
    print np.median(rain_types3[(rain_types3>=0)&(rain_types3<1000)])
    plt.hist(rain_types3[(rain_types3>=0)&(rain_types3<1000)])
    plt.show()
    '''	
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
#0.00898616252983 --
#0.00894938332555
#0.00894852729502
#0.00894846604764
#0.00894788853756
#0.00894671310461
#0.00894636668274
#0.00894535250385
#0.0089344109825
#0.00893531092568
#0.00898349408228

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
#0.00907174536772 ***
#0.00907146849158 -> 0.00849297
#0.00907144426707 -> 0.00849229
#0.00910805136467 -> 0.00846733

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

