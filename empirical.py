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
    for i, v in enumerate(predictions):
	 if not is_cdf_valid(v):
		print v
    obscdf = np.array([heaviside(i) for i in actuals])
    crps = np.mean(np.mean((predictions - obscdf) ** 2))
    return crps

def step(center, length=70):
	x = [1.]*length
	for i in range(0, int(center)):
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

def hmdir(times, rr, w, x, d, ey, defaults):
	valid_t = times[(rr>=0)&(rr<100)]
	valid_r = rr[(rr>=0)&(rr<100)]
	valid_r = valid_r
	if len(valid_t)==0: return 0	
	if len(valid_t)<2: return valid_r[0]/60.
	f = interpolate.interp1d(valid_t, valid_r)
	ra = range(int(valid_t.min()), int(valid_t.max()+1))	
	tl = f(ra)
	#plt.plot(tl)

	k = 5
	if len(tl)>=k:
		tl = scipy.signal.savgol_filter(tl, k, 3)
		
	#plt.plot(tl)
	#plt.show()
	est = sum(tl)/60.
	return est
	
	
def hmdirs(times, rr, w, hts, distances, ey, defaults):
	hour = [0.]*61
	for i in range(1, len(times)):
		for j in range(int(times[len(times)-i]), int(times[len(times)-i-1])):
			v = rr[len(times)-i-1]
			q = w[len(times)-i-1]
			ht = hts[len(times)-i-1] 
			if v>=0 and v<100 and not ht in [6, 8]:
				hour[j]=v			
			
	est = sum(hour)/60.
	return est
	
def all_good_estimates(rr, distances, radar_indices, w, times, hts, ey, defaults, compos):
	age = []
	agd = []
	for radar in radar_indices:
		rain = rr[radar]
		q = w[radar]
		est = hmdir(times[radar], rr[radar], w[radar], hts[radar], distances[radar], ey, defaults)
		age.append(est)
		agd.append(distances[radar][0])
	return age, agd


def mean(x, default=0):
	if len(x)==0: return default
	return np.mean(x)


def is_cdf_valid(case):
    if case[0] < 0 or case[0] > 1:
        return False
    for i in xrange(1, len(case)):
        if (case[i] - 1)>1e-3:
	    print case[i]-1.
            return False
	if (case[i-1] - case[i])>1e-3:
	    print case[i-1], case[i]
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


def radar_features(rr, hts, w, d, waters, composites):

	m = float(len(rr))
	composite_neg_rate = len(composites[(composites!=-99900)&(composites<0)])/m

	error_rate = len(rr[rr<0])/m
	zero_rate = len(rr[rr==0])/m
	oor_rate = len(rr[rr>2000])/m
	rain_rate = len(rr[(rr>10)&(rr<=100)])/m

	bad_q = len(w[w==0])/m
	oor_q = len(w[w>1])/m
	good_q = len(w[w==1])/m
	ok_q = len(w[(w>0)&(w<1)])/m

	distance = d[0]
	ht0 = len(hts[hts==0])
	ht1 = len(hts[hts==1])
	ht2 = len(hts[hts==2])
	ht3 = len(hts[hts==3])
	ht4 = len(hts[hts==4])
	ht5 = len(hts[hts==5])

	ht6 = len(hts[hts==6])
	ht7 = len(hts[hts==7])
	ht8 = len(hts[hts==8])

	ht9 = len(hts[hts==9])
	ht13 = len(hts[hts==13])
	ht14 = len(hts[hts==14])

	return [composite_neg_rate, ht13/m, np.sqrt(ok_q), oor_q, ht6/m, ht2/m, m, -1]	

#0.00895660879826
#0.00895629729627
#0.0089555104762
#0.00894224941284
#0.00893217139966
#0.00893150685717
#0.00892673459363
#0.00892397448774
#0.00892321039254
#0.00892310908378
#0.00892309486393

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
    water_ind = header.index('LogWaterVolume')
    mwm_ind = header.index('MassWeightedMean')


    y = []
    ids = []
    avgs = []
    cX = []	
    for i, row in enumerate(reader):
	ids.append(row[id_ind])	        
	times = parse_floats(row, time_ind)
	distances = parse_floats(row, distance_ind)
        rr1 = parse_rr(row, rr1_ind)
        rr2 = parse_rr(row, rr2_ind)
        rr3 = np.fabs(parse_rr(row, rr3_ind))
	w = parse_floats(row, rad_q_ind)
	hidro_types = parse_floats(row, hydro_type_ind)
	waters = parse_floats(row, water_ind)	
	mwms = parse_floats(row, mwm_ind)
	composites = parse_floats(row, composite_ind)

	if expected_ind >= 0:
		ey = float(row[expected_ind])
		y.append(ey)
	else:
		ey = -1
	
	radar_indices = split_radars(distances, times)


	rr1_estimates, rr1_d = all_good_estimates(rr1, distances, radar_indices, w, times, hidro_types, ey, [0.33, 33.31, 33.31], composites)	
	rr2_estimates, rr2_d  = all_good_estimates(rr2, distances, radar_indices, w, times, hidro_types, ey, [1.51, 36.37, 81.17], composites)
	rr3_estimates, rr3_d  = all_good_estimates(rr3, distances, radar_indices, w, times, hidro_types, ey, [4.52, 38.60, 42.34], composites)

	avgs.append([mean(rr1_estimates, -1), mean(rr2_estimates, -1), mean(rr3_estimates, -1)])
	'''
	radar_f = []
	for radar in radar_indices:
		rf = radar_features(rr1[radar], hidro_types[radar], w[radar], distances[radar], waters[radar], composites[radar])
		radar_f.append(rf)
			
	total = np.mean(radar_f, axis=0)	
	total[-1]=mean(rr1_estimates)
	cX.append(total)
	'''
	
	if i % 10000 == 0:
		print "Completed row %d" % i
    
    return ids, np.array(avgs), np.array(y), np.array(cX)

def cap_mean(a):
	rr = []
	for x in a:
		if x<0:
			x=0
		if x>70:
			x=70
		rr.append(x)
	return np.mean(rr)

def as_labels(y):
	labels = np.array([1]*len(y))
	for i, yi in enumerate(y):
		if yi == 0:
			labels[i]=0

	return labels

def split(X, y):
	from sklearn.cross_validation import StratifiedShuffleSplit
	labels = as_labels(y)
	sss = StratifiedShuffleSplit(labels, 1, test_size=0.3)
	for a, b in sss:
		train_X = X[a]
		val_X = X[b] 
			
		train_y = y[a]
		val_y = y[b]
		
		train_labels = labels[a]
		val_labels = labels[b]		
		
	return train_X, train_y, val_X, val_y, train_labels, val_labels 


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
#0.00899329563108
#0.00896273995689
#0.00895327069295
#0.00895322370697
#0.00895317650512
#0.00893217139966


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
#0.00908048569227 -> 0.00844376
#0.00907063607574
#0.00907063537653
#0.00904911201717 => 0.00842818
#0.00904491469844

#Baseline CRPS: 0.00965034244803
#1126695 training examples
#987398 0s
#133717 valid no 0
#5580 invalid

def fit_cdf(X, y):
	ty = []
	for yi in y:
		if yi>70: 
			ty.append(70)
		else: 
			ty.append(yi)
	ty = np.array(ty)
	

	n=70+2
	m=70
	models = []
	for x in X.T:
		q = scipy.stats.mstats.mquantiles(x[x>0.1], np.arange(0,n-1)/float(n-2))
		breaks = np.concatenate(([-1, 0], q))
		model = np.zeros((n,m))
		for i in range(0, n):
			d = ty[(x>breaks[i])&(x<=breaks[i+1])]
			h, _ = np.histogram(d, bins=range(0, m+1)	)
			for j in range(1, len(h)):
				h[j]+=h[j-1]

			model[i]=h/float(h[-1])
		models.append(model)
	return models

def predict_cdf(X, models):
	n=70+2
	m=70
	predictions = np.zeros((len(X), 70))	
	for j, x  in enumerate(X.T):
		q = scipy.stats.mstats.mquantiles(x[x>0.1], np.arange(0,n-1)/float(n-2))
		breaks = np.concatenate(([-1, 0], q))
		model = models[j]			
		for i in range(0, n):
			predictions[(x>breaks[i])&(x<=breaks[i+1]),0:m]+=model[i]/float(len(X.T))
	return predictions

def classification_features(cX, predictions):
	a = []
	for i in range(0, len(cX)):
		b = []
		b.extend(cX[i])
		b.extend(predictions[i])
		a.append(b)
	return np.array(a)


_, X, y, cX= data_set('train_2013.csv')

model = fit_cdf(X, y)
predictions = predict_cdf(X, model)
print 'CRPS',  calc_crps(predictions, y) 

'''
cX = classification_features(cX, predictions)

from sklearn import svm
clf = svm.LinearSVC(verbose=3, dual=False)
from sklearn import linear_model
clf = linear_model.LogisticRegression(tol=1e-8, C=128)
#clf = svm.SVC(verbose=3, probability=True)
y_labels = as_labels(y)
clf.fit(cX, y_labels)

print 'Training Accuracy', clf.score(cX, y_labels)

from sklearn.metrics import classification_report
print classification_report(y_labels, clf.predict(cX))

X_l = clf.predict_proba(cX)
for i, l in enumerate(X_l):
	if predictions[i][0] >0.95:
		predictions[i]=(predictions[i]+1)/2

print 'CRPS: ',  calc_crps(predictions, y)
'''


print 'Predicting for sumbission...'
print 'Loading test file...'
ids, X, y, cX= data_set('test_2014.csv')

predictions = predict_cdf(X, model)


'''
cX = classification_features(cX, predictions)

X_l = clf.predict_proba(cX)
for i, l in enumerate(X_l):
	if l[0] >0.5:
		predictions[i]=(predictions[i]+1)/2
'''

cdfs = predictions

print 'Writing submision file...'

writer = csv.writer(open('classifier-cdfavg-sub.csv', 'w'))
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

