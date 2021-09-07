function R = get_max_entropy_model(EW,ncg)
% Compute the probability of spiking for neuronal groups of length ncg 
% selected from a matrix EW by using a maximum entropy model of orders 1 
% (independent model) and 2;
% Input
% EW: binary matrix (-1,1) of dimensions (nr_bins x nr_units); 
% nr_bins = T/binw, where T is the time length the of the continuous record, 
% and binw is the bin width of the binned spiking activity. The content of 
% a bin is -1 for no spike in that time bin and 1 for one or more spikes in
% that time bin; typical binw = 20 msec; nr_units is number of neurons in 
% the population 
% ncg: number of neurons in a group - word length for max entropy modeling
% Output
% R.Pn: probability of spiking for experimental data
% R.ME1: structure with P1, H, J for 1st order model (P - probability, H - 
% local fields, J - exchange interactions in the max entropy model)
% R.ME2: structure with P2, H, J for 2nd order model
% R.Perf: structure with performance indexes (f1 - fraction of explained 
% entropy f2 = (S1 - S2)/(S1 - SN); f2 = (D1 - D2)/D1; see Tang et al, 
% J. Neurosci, 2008 

% Implementation from Tang et al, J. Neurosci, 2008 with a modification 
% that improve the convergence speed; Function uses parfor loop for faster 
% computution - one needs Parallel Processing Toolbox 

if(nargin < 2)
    ncg = 6;
end

if(isempty(EW))
    disp('No input data');
    R = []; return;
end

[nr_bins,nr_max_neurons] = size(EW);    
nr_neurons = ncg;
nr_states = 2^nr_neurons;

% nr_groups: number of groups ncg neurons that are modeled with max entropy
Cg = combnk(1:nr_max_neurons,nr_neurons); 
nr_groups = size(Cg,1);  

% limit the number of groups
nr_max_groups = 5000;   % equivalent of Comb(15,6) - number of groups of 6 
% for a population of 15 units
if(nr_groups > nr_max_groups)
    rng('shuffle'); 
    rind = randi(nr_groups,[nr_groups,1]);
    IX = rind(1:nr_max_groups); 
    C = Cg(IX,:); 
    nr_groups = size(C,1);
else
    C = Cg;
end

% parameters for maximum entropy modeling
alpha = 0.1;    
nr_max_iter = 50000; 
lc = 0.1;   % level of change in local fields and interactions in percents

% compute Pn,P1 and P2 for all groups
Pn = zeros(nr_groups,nr_states); 
P1 = zeros(nr_groups,nr_states); 
H1 = cell(nr_groups,1); J1 = cell(nr_groups,1);
P2 = zeros(nr_groups,nr_states);
H2 = cell(nr_groups,1); J2 = cell(nr_groups,1);
Prf = cell(nr_groups,1);
% parfor loop
parfor k = 1:nr_groups
    EWg = zeros(nr_bins,nr_neurons);
    for j = 1:nr_neurons
        EWg(:,j) = EW(:,C(k,j));    %#ok<PFBNS>
    end
    Pn(k,:) = get_experimental_probability(EWg,nr_bins,nr_states);
    [P1(k,:),H1{k},J1{k},ni1] = max_entropy_one_probability(EWg,nr_neurons,...
        nr_states,alpha,nr_max_iter,lc);
    [P2(k,:),H2{k},J2{k},ni2] = max_entropy_two_probability(EWg,nr_bins,...
            nr_neurons,nr_states,alpha,nr_max_iter,lc);
    Prf{k} = get_performance_indexes(P1(k,:),ni1,P2(k,:),ni2,Pn(k,:));
    fprintf('group %d\n',k);
end

% Select the groups for which S1 > S2 > SN
K = 0;
for i = 1:nr_groups
    if((Prf{i}.S1 > Prf{i}.S2) && (Prf{i}.S2 > Prf{i}.Sn))
        K = K + 1;
        Pn(K,:) = Pn(i,:); 
        P1(K,:) = P1(i,:); H1{K} = H1{i}; J1{K} = J1{i};
        P2(K,:) = P2(i,:); H2{K} = H2{i}; J2{K} = J2{i};
        Prf{K} = Prf{i}; 
    end
end
if(K == 0)
    disp('No groups with S1 > S2 > SN');
    R = []; return;
end

Pn = Pn(1:K,:);
P1 = P1(1:K,:); H1 = H1(1:K); J1 = J1(1:K);
P2 = P2(1:K,:); H2 = H2(1:K); J2 = J2(1:K); 
Prf = Prf(1:K); 
nr_groups = K;

% get mean performance
f1 = zeros(nr_groups,1);
f2 = zeros(nr_groups,1);
for k = 1:nr_groups
    f1(k) = Prf{k}.f1;
    f2(k) = Prf{k}.f2;
end
mf1 = mean(f1); mf2 = mean(f2);

ME1.P = P1; ME1.H = H1; ME1.J = J1;
ME2.P = P2; ME2.H = H2; ME2.J = J2;
Perf.Prf = Prf;
Perf.f1 = f1; Perf.mf1 = mf1;
Perf.f2 = f2; Perf.mf2 = mf2;

R.Pn = Pn; R.ME1 = ME1; R.ME2 = ME2; R.Perf = Perf;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Compute the probability of data EW (Pn)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Pn = get_experimental_probability(EW,nr_bins,nr_states)
Y = zeros(nr_bins,1);
for i = 1:nr_bins
    bw = entw2binw(EW(i,:));
    Y(i) = binw2int(bw);
end
Pn = zeros(1,nr_states);
for i = 1:nr_states
    IX = find(Y == i-1);
    if(isempty(IX))
        Pn(i) = 0;
    else
        Pn(i) = length(IX)/nr_bins;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Compute max. entropy probability of independent spiking (P1) from EW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [P1,Hc,Jc,nr_iter] = max_entropy_one_probability(EW,nr_neurons,nr_states,...
    alpha,nr_max_iter,lc)
[FR,I2] = get_independent_interactions(EW,nr_neurons);
Hc = FR; Jc = I2;
Pc = get_max_entropy_probability(Hc,Jc,nr_neurons,nr_states);
alphac = alpha; 
nr_iter = 0; 
while(nr_iter < nr_max_iter)
    [Hn,Jn,nc] = update_local_fields_interactions(Pc,Hc,Jc,FR,I2,nr_neurons,...
        nr_states,alphac,lc);
    if(nc == true) 
        P1 = Pc; break;
    end
    P1 = get_max_entropy_probability(Hn,Jn,nr_neurons,nr_states);
    Pc = P1; Hc = Hn; Jc = Jn;
    nr_iter = nr_iter+1;
    alphac = alphac - alpha/(nr_max_iter + 1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Compute max. entropy probability of order 2 (P2) from EW 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [P2,Hc,Jc,nr_iter] = max_entropy_two_probability(EW,nr_bins,nr_neurons,...
    nr_states,alpha,nr_max_iter,lc)
[FR,I2] = get_data_firing_rate_interactions(EW,nr_bins,nr_neurons);
Hc = FR; Jc = I2;
P2i = get_max_entropy_probability(Hc,Jc,nr_neurons,nr_states);
Pc = P2i;
alphac = alpha; 
nr_iter = 0;
while(nr_iter < nr_max_iter)
    [Hn,Jn,nc] = update_local_fields_interactions(Pc,Hc,Jc,FR,I2,nr_neurons,...
        nr_states,alphac,lc);
    if(nc == true)
        P2 = Pc; break;
    end
    P2 = get_max_entropy_probability(Hn,Jn,nr_neurons,nr_states);
    Pc = P2; Hc = Hn; Jc = Jn; 
    nr_iter = nr_iter+1;
    alphac = alphac - alpha/(nr_max_iter + 1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Compute the performance indexes for maximum entropy modeling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Perf = get_performance_indexes(P1,n1,P2,n2,Pn)
Perf.nr_iter1 = n1; Perf.nr_iter2 = n2;
S1 = get_entropy(P1'); Perf.S1 = S1;
S2 = get_entropy(P2'); Perf.S2 = S2;
Sn = get_entropy(Pn'); Perf.Sn = Sn;
In = S1 - Sn; Perf.In = In;
I2 = S1 - S2; Perf.I2 = I2;
f1 = I2/In; Perf.f1 = f1;
D1 = get_KLB_div(Pn,P1); Perf.D1 = D1;
D2 = get_KLB_div(Pn,P2); Perf.D2 = D2;
f2 = (D1 - D2)/D1; Perf.f2 = f2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Auxiliary functions for max entropy modeling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [FR,I2] = get_independent_interactions(EW,nr_neurons)
% compute the firing rates FR and second order interactions I2 from data
FR = mean(EW); FR = FR';
I2 = zeros(nr_neurons,nr_neurons);
for i = 1:nr_neurons
    for j = 1:nr_neurons
        I2(i,j) = FR(i)*FR(j);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [FR,I2] = get_data_firing_rate_interactions(EW,nr_bins,nr_neurons)
% compute the firing rates FR and second order interactions I2 from data
FR = mean(EW); FR = FR';
I2 = zeros(nr_neurons,nr_neurons);
for i = 1:nr_neurons
    for j = 1:nr_neurons
        for t = 1:nr_bins
            I2(i,j) = I2(i,j) + EW(t,i)*EW(t,j);
        end
    end
end
I2 = I2/nr_bins;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function P = get_max_entropy_probability(H,J,nr_neurons,nr_states)
% compute the energy state for all states
E = zeros(nr_states,1);
for k = 1:nr_states
    bw = int2binw(k-1,nr_neurons); 
    S = binw2ent(bw);           
    for i = 1:nr_neurons
        E(k) = E(k) - H(i)*S(i);
        for j = 1:nr_neurons
            if(i == j)
                continue;
            end
            E(k) = E(k) - (J(i,j)*S(i)*S(j))/2;
        end
    end
end
% compute maximum entropy distribution
P = zeros(1,nr_states);
SumP = 0;
for k = 1:nr_states
    P(k) = exp(-E(k));
    SumP = SumP + P(k);
end
P = P/SumP;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Hn,Jn,nc] = update_local_fields_interactions(P,Hc,Jc,FR,I2,...
    nr_neurons,nr_states,alpha,lc)
% P: current model probability
% Hc,Jc: current model coefficents
% Hn, Jn: next model coefficients
% nc: true  change in local files and interactions < level of change lc
% m_chg: maximum change in local filed and interactions
% compute current firing rate FRm and 2nd order interactions I2m
FRm = zeros(nr_neurons,1);
for i = 1:nr_neurons
    for Vj = 1:nr_states
        bw = int2binw(Vj-1,nr_neurons); 
        Sj = binw2ent(bw);
        FRm(i) = FRm(i) + Sj(i)*P(Vj);
    end
end
I2m = zeros(nr_neurons,nr_neurons);
for i = 1:nr_neurons
    for j = 1:nr_neurons
        if(i == j)
            continue;
        end
        for Vk = 1:nr_states
            bw = int2binw(Vk-1,nr_neurons);
            Sk = binw2ent(bw);
            I2m(i,j) = I2m(i,j) + Sk(i)*Sk(j)*P(Vk);
        end
    end
end
Hn = zeros(nr_neurons,1);
Jn = zeros(nr_neurons,nr_neurons);
for i = 1:nr_neurons
    Hn(i) = Hc(i) + alpha*sign(FR(i))*log(abs(FR(i)/FRm(i)));
    for j = 1:nr_neurons
        if(i == j)
            continue;
        end
        Jn(i,j) = Jc(i,j) + alpha*sign(I2(i,j))*log(abs(I2(i,j)/I2m(i,j)));
    end
end
% check the change for the cell interactions
[nc,~] = check_change_local_fields_interactions(Hc,Jc,Hn,Jn,nr_neurons,lc); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [nc,max_chg] = check_change_local_fields_interactions(Hc,Jc,Hn,Jn,...
    nr_neurons,lc)
nc = false;
ch_Hc = 100*abs((Hc - Hn)./Hc);
ch_Jc = zeros(nr_neurons,nr_neurons);
for i = 1:nr_neurons
    ch_Jc(i,i) = 0;
end
for i = 1:nr_neurons
    for j = 1:nr_neurons
        if(i == j)
            continue;
        else
            ch_Jc(i,j) = 100*abs((Jc(i,j) - Jn(i,j))/Jc(i,j));
        end
   end
end
m_chg = [max(ch_Hc); max(max(ch_Jc))];
if(m_chg(1) < lc && m_chg(2) < lc)
    nc = true;
end
max_chg = max(m_chg);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Transformations  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function bw = int2binw(x,N)
% transform an integer in cod binar
bw = zeros(1,N);
if(x >= 2^(N-1))
    bw(1) = 1;
end
x = x - bw(1)*2^(N-1);
for k = 2:N
    if(x >= 2^(N-k))
        bw(k) = 1;
    end
    x = x - bw(k)*2^(N-k);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = binw2int(bw)
% transform a binary word to an integer
N = size(bw,2); y = 0;
for k = 1:N
    y = y + bw(k)*2^(N-k);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ew = binw2ent(bw)
% transform a binary word into an entropy word
N = size(bw,2);
ew = zeros(1,N);
for k = 1:N
    if(bw(k) == 1)
        ew(k) = 1;
    else
        ew(k) = -1;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function bw = entw2binw(ew)
% transform an entropy word with -1|+1 bits into binary word
N = size(ew,2);
bw = zeros(1,N);
for k = 1:N
    if(ew(k) == 1)
        bw(k) = 1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Function for performance computing 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function S = get_entropy(P)
% P must be column vector
P = P(P~=0);
S = -P'*log2(P);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function D = get_KLB_div(P1,P2)
% compute thr Kullback_Leibler divergence
D = 0;
for i = 1:length(P1)
    if(P1(i) == 0 || P2(i) == 0)
        continue;
    end
    D = D + P1(i)*log2(P1(i)/P2(i));
end