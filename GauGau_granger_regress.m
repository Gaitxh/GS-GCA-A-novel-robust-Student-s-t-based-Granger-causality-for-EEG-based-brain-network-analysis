function [ret] = GauGau_granger_regress(X,nlags,STATFLAG)


% figure regression parameters
nobs = size(X,2);
nvar = size(X,1);
if(nvar>nobs) error('error in cca_granger_regress: nvar>nobs, check input matrix'); end
if nargin == 2, STATFLAG = 1; end

% remove sample means if present (no constant terms in this regression)
m = mean(X');
if(abs(sum(m)) > 0.0001)
    mall = repmat(m',1,nobs);
    X = X-mall;
end

% construct lag matrices
lags = -999*ones(nvar,nobs-nlags,nlags);
for jj=1:nvar
    for ii=1:nlags
        lags(jj,:,nlags-ii+1) = X(jj,ii:nobs-nlags+ii-1);
    end
end

%  unrestricted regression (no constant term)
regressors = zeros(nobs-nlags,nvar*nlags);
for ii=1:nvar,
    s1 = (ii-1)*nlags+1;
    regressors(:,s1:s1+nlags-1) = squeeze(lags(ii,:,:));
end

for ii=1:nvar
    xvec = X(ii,:)';
    xdep = xvec(nlags+1:end);   
    [beta(:,ii),err] = GauGau_Estimate(regressors, xdep);   
    xpred(:,ii) = regressors*beta(:,ii);  % keep hold of predicted values
    u(:,ii) = xdep-xpred(:,ii);
    RSS1(ii) = err;
    C(ii) = err; 
end
covu = cov(u);

%   A rectangular matrix A is rank deficient if it does not have linearly independent columns.
%   If A is rank deficient, the least squares solution to AX = B is not unique.
%   The backslash operator, A\B, issues a warning if A is rank deficient and
%   produces a least squares solution that has at most rank(A) nonzeros.

%   restricted regressions (no constant terms)
for ii=1:nvar
    xvec = X(ii,:)';
    xdep = xvec(nlags+1:end);          % dependent variable
    caus_inx = setdiff(1:nvar,ii);     % possible causal influences on xvec
    u_r = zeros(nobs-nlags,nvar,'single');
    for jj=1:length(caus_inx)
        eq_inx = setdiff(1:nvar,caus_inx(jj));  % vars to include in restricted regression (jj on ii)
        regressors = zeros(nobs-nlags,length(eq_inx)*nlags);
        for kk=1:length(eq_inx)
            s1 = (kk-1)*nlags+1;
            regressors(:,s1:s1+nlags-1) = squeeze(lags(eq_inx(kk),:,:));
        end
        [beta_r,err] = GauGau_Estimate(regressors, xdep);
        temp_r = xdep-regressors*beta_r;
        RSS0(ii,caus_inx(jj)) = err;
        S(ii,caus_inx(jj)) = err; % dec 08
        u_r(:,caus_inx(jj)) = temp_r;
    end
    covr{ii} = cov(u_r);
    
end
% calc Granger values
gc = ones(nvar).*NaN;
doi = ones(nvar).*NaN;
%   do Granger f-tests if required
if STATFLAG == 1,
    prb = ones(nvar).*NaN;
    ftest = zeros(nvar);
    n2 = (nobs-nlags)-(nvar*nlags);
    for ii=1:nvar-1
        for jj=ii+1:nvar
            ftest(ii,jj) = ((RSS0(ii,jj)-RSS1(ii))/nlags)/(RSS1(ii)/n2);    % causality jj->ii
            prb(ii,jj) = 1 - cca_cdff(ftest(ii,jj),nlags,n2);
            
            ftest(jj,ii) = ((RSS0(jj,ii)-RSS1(jj))/nlags)/(RSS1(jj)/n2);    % causality ii->jj
            prb(jj,ii) = 1 - cca_cdff(ftest(jj,ii),nlags,n2);
            
            gc(ii,jj) = log(S(ii,jj)/C(ii));
            gc(jj,ii) = log(S(jj,ii)/C(jj));
            doi(ii,jj) = gc(ii,jj) - gc(jj,ii);
            doi(jj,ii) = gc(jj,ii) - gc(ii,jj);
        end
    end
else
    ftest = -1;
    prb = -1;
    for ii=1:nvar-1,
        for jj=ii+1:nvar,
            gc(ii,jj) = log(S(ii,jj)/C(ii));
            gc(jj,ii) = log(S(jj,ii)/C(jj));
            doi(ii,jj) = gc(ii,jj) - gc(jj,ii);
            doi(jj,ii) = gc(jj,ii) - gc(ii,jj);
        end
    end
end

%   do r-squared and check whiteness, consistency
if STATFLAG == 1,
    df_error = (nobs-nlags)-(nvar*nlags);
    df_total = (nobs-nlags);
    for ii = 1:nvar
        xvec = X(ii,nlags+1:end);
        rss2 = xvec*xvec';
        rss(ii) = 1 - (RSS1(ii) ./ rss2);
        rss_adj(ii) = 1 - ((RSS1(ii)/df_error) / (rss2/df_total) );
        %waut(ii) = cca_whiteness(X,u(:,ii));
        waut(ii) = -1;  % TEMP APR 19 COMPILER ERROR
    end
    cons = cca_consistency(X,xpred);
else
    rss = -1;
    rss_adj = -1;
    waut = -1;
    cons = -1;
end

%   organize output structure
ret.gc = gc;
ret.prb = prb;


