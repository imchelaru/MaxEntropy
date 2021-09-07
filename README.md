# MaxEntropy
The function R = get_max_entropy_model(EW,ncg) 
compute the probability of spiking for neuronal groups of length ncg 
selected from a matrix EW by using a maximum entropy model of orders 1 
(independent model) and 2;
Input
EW: binary matrix (-1,1) of dimensions (nr_bins x nr_units); 
nr_bins = T/binw, where T is the time length the of the continuous record, 
and binw is the bin width of the binned spiking activity. The content of 
a bin is -1 for no spike in that time bin and 1 for one or more spikes in
that time bin; typical binw = 20 msec; nr_units is number of neurons in 
the population 
ncg: number of neurons in a group - word length for max entropy modeling
Output
R.Pn: probability of spiking for experimental data
R.ME1: structure with P1, H, J for 1st order model (P - probability, H - 
local fields, J - exchange interactions in the max entropy model)
R.ME2: structure with P2, H, J for 2nd order model
R.Perf: structure with performance indexes (f1 - fraction of explained 
entropy f2 = (S1 - S2)/(S1 - SN); f2 = (D1 - D2)/D1; see Tang et al, 
J. Neurosci, 2008 
