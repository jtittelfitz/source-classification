%
%
%� CMAP, Ecole Polytechnique 
%contributors: Joan Bruna and St�phane Mallat (2010)
%bruna@cmap.polytechnique.fr
%
%This software is a computer program whose purpose is to implement
%an image classification architecture based on the Scattering Transform.
%
%This software is governed by the CeCILL  license under French law and
%abiding by the rules of distribution of free software.  You can  use, 
%modify and/ or redistribute the software under the terms of the CeCILL
%license as circulated by CEA, CNRS and INRIA at the following URL
%"http://www.cecill.info". 
%
%As a counterpart to the access to the source code and  rights to copy,
%modify and redistribute granted by the license, users are provided only
%with a limited warranty  and the software's author,  the holder of the
%economic rights,  and the successive licensors  have only  limited
%liability. 
%
%In this respect, the user's attention is drawn to the risks associated
%with loading,  using,  modifying and/or developing or reproducing the
%software by the user in light of its specific status of free software,
%that may mean  that it is complicated to manipulate,  and  that  also
%therefore means  that it is reserved for developers  and  experienced
%professionals having in-depth computer knowledge. Users are therefore
%encouraged to load and test the software's suitability as regards their
%requirements in conditions enabling the security of their systems and/or 
%data to be ensured and,  more generally, to use and operate it in the 
%same conditions as regards security. 
%
%The fact that you are presently reading this means that you have had
%knowledge of the CeCILL license and that you accept its terms.
%
%
function out=generic_classifier(options)
%out=generic_classifier(options)
%Function implementing the classification architecture
%described in the paper "Classification with Scattering Operators"
%
%
%options.set= (['mnist'],'usps','rmnist','caltech','brodatz','cure','yaleb','synthetic') 
options.set = 'approx_speed';
%options.Ltrain: number of training samples per class [200]
%options.Ltest:number of testing samples per class [whole set]
%options.J scattering max scale[3]
%options.M scattering max order[2]
%options.L number of orientations [6]
%options.downsample=([2],1,0) twice oversampling
%options.Dmax [256] maximum dimension of affine spaces
%options.singlespace ([0],1) use the same spaces for all classes 
%options.fixed_dim [-2] : use projection spaces of fixed dimension (optimized
%during cross-validation)
%                    -1 : use penalized model selection algorithm (penalization
%                    parameter optimized during cross-validation)
%                    k  (k>=0) : use projection spaces of dimension k (in
%                    particular, k==0 reduces to centroid-prototype
%                    classification)

%options.validation_set_size [0.2] size of validation set used for learning penalty
%options.svm:[0] scattering+PCA classifier
%			 1   scattering+svm (requires SimpleSVM* package)
%options.nn: [0]
%			 1   scattering+nearest neighbors classifier
%
%output contains the following fields:
%out.erate: error rate on test set;
%out.eratev: error rate on the validation set;
%out.lambda: best estimated penalization factor;
%out.eigenvects: estimated eigenvectors for each class;
%out.eigenvalues: estimated eigenvalues for each class;
%out.moyenne: estimated average for each class;
%out.variance: estimated variance for each class;
%out.classifier: the classifier output for the testing set;
%out.ncomps: average number of approximation dimensions;
%out.aprerr: average approximation error; 
%out.ncomps_or: average number of intra-class approximation dimensions;
%out.aprerr_or : average intra-class approximation error;
%
%%
%(*) The SimpleSVM package can be downloaded at
% http://sourceforge.net/projects/simplesvm/
%

%%%read options
%dataset parameters
set= getoptions(options,'set','mnist');
Ltrain=getoptions(options,'Ltrain',200);
Ltest=getoptions(options,'Ltest',9999);
maxclasses=getoptions(options,'maxclasses',0);
arot=getoptions(options,'arot',0);
adil=getoptions(options,'adil',0);
switch lower(set)
	case 'mnist'
		[train,test]=retrieve_mnist_data(Ltrain,Ltest,maxclasses);
	case 'mnist_rotdil'
		[train,test]=retrieve_mnist_rotdil(Ltrain,Ltest,9,arot,adil);
	case 'mnist_rotation'
		[train,test]=retrieve_mnist_rotate(Ltrain,Ltest,9);
	case 'usps'
		[train,test]=retrieve_usps_data(Ltrain,Ltest);
	case 'rmnist'
		[train,test]=retrieve_rmnist_data(Ltrain,Ltest);
	case 'caltech'
		[train,test]=retrieve_caltech_data(Ltrain,Ltest,101,0);
	case 'scenes'
		[train,test]=retrieve_scenes_data(Ltrain,Ltest);
	case 'brodatz'
		[train,test]=retrieve_brodatz_data(Ltrain,Ltest);
	case 'brodatz_rotation'
		[train,test]=retrieve_brodatz_rot_data(Ltrain,Ltest);
	case 'cure'
		[train,test]=retrieve_cure_data(Ltrain,Ltest,maxclasses);
	case 'yaleb'
		[train,test]=retrieve_yaleb(Ltrain,Ltest);
	case 'synthetic'
		[train,test]=retrieve_synthetic_texture(Ltrain,Ltest);
    case 'approx_speed'
        [train,test]=retrieve_train_data(Ltrain,Ltest);
	otherwise
		error('unknown database')
end
imsize = size(train{1}{1});


%scattering parameters
J = getoptions(options,'J',3);
M = getoptions(options, 'M', 2);
L = getoptions(options, 'L', 8);
[W,Phi] = gaborwavelets(J,L,imsize);
options.W = W;
options.Phi = Phi;
if isfield(options,'combined')
      if options.combined >0
        options.L=8;
	options.Jrot=getoptions(options,'Jrot',3);
	options.Wrot=rotationwavelets(L,options.Jrot);
      end
end
if isfield(options,'format')==0
	options.format = 'array';
end

%classifier parameters
Dmax = getoptions(options,'Dmax',256);
Dmax=min(Dmax,Ltrain);
singlespace=getoptions(options,'singlespace',0);
fixed_dim = getoptions(options,'fixed_dim',-2);
%percentage of the training samples reserved for the learning of the parameter Lambda
alpha =getoptions(options,'validation_set_size',0.2);
alpha=1-alpha;
svm=getoptions(options,'svm',0);
nn=getoptions(options,'nn',0);
maha=getoptions(options,'maha',0);


%%%%%%%%%TRAINING%%%%%%%%%%%
D = size(train,2);

if svm==1

  for d=1:D
    total_train(d) = size(train{d},2);
    auxiliar = rand(1,total_train(d));
    [res,order{d}]=sort(auxiliar);
    pca_train(d) = round(alpha*size(train{d},2));
  end


  count=1;
  tcount=1;
  for d=1:D
        prevcount=count;
	fprintf('doing class %d Training \n',d-1)
	total_train(d)=size(train{d},2);
	for s=1:pca_train(d)
		strain(:,count)=scatt(train{d}{s},options);
		train_label(count)=d-1;
		count=count+1;
	end
	%total_test(d)=size(test{d},2);
	total_test(d)=total_train(d)-pca_train(d);%size(test{d},2);
	fprintf('doing class %d Test \n',d-1)
	for s=1:total_test(d)
		%stest(:,tcount)=scatt(test{d}{s},options);
		stest(:,tcount)=scatt(train{d}{pca_train(d)+s},options);
		test_label(tcount)=d-1;
		tcount=tcount+1;
	end
        variance(d) = sum(var(strain(:,prevcount:count-1),1,2));
  end

  %cross-validation
  varikern=2.^[-2:7];
  for k=1:length(varikern)
  [perf(k),suppvects]=svm_generic_classifier(strain,train_label,stest,test_label,varikern(k)*mean(variance(:)));
  end

  [maxperf,maxvari]=max(perf);

  %real testing
  clear stest;
  clear test_label;
  tcount=1;
  for d=1:D
	total_test(d)=size(test{d},2);
	fprintf('doing class %d Test \n',d-1)
	for s=1:total_test(d)
		stest(:,tcount)=scatt(test{d}{s},options);
		test_label(tcount)=d-1;
		tcount=tcount+1;
	end
  end

  [fperf,suppvects]=svm_generic_classifier(strain,train_label,stest,test_label,varikern(maxvari)*mean(variance(:)));

  out.eratev=1-maxperf;
  out.erate=1-fperf;
  out.ncomps = suppvects;

else
  Ltrain = size(train{1},2);
  Ltest = size(test{1},2);

  for d=1:D
    total_train(d) = size(train{d},2);
    auxiliar = rand(1,total_train(d));
    [res,order{d}]=sort(auxiliar);
    pca_train(d) = round(alpha*size(train{d},2));
    if svm==2
      pca_train(d) = total_train(d);
    end
  end

  for d=1:D
    fprintf('doing class %d (',d-1)
    for s=1:pca_train(d)
	if (mod(s,round(Ltrain/10))==1)
          fprintf(' %d,',s)
        end
          descr{d}(s,:) = (scatt(train{d}{s},options))';
    end
    moyenne{d} = mean(descr{d},1);
    variance{d} = var(descr{d},1,1);
    fprintf(' size is (%d,%d) \n',size(descr{d},1), size(descr{d},2))
  end

  if nn==1
  %nearest neighbor classifier
  fprintf('trying nn classifier...\n')
  l=1;
  maxtrainsize=max(total_train);
  erate=0;
  for d=1:D
     for s=1:size(test{d},2)
        lab = d;
        [slice]=scatt(test{d}{s},options)';
        slicemat=ones(maxtrainsize,1)*slice;
        for dd=1:D
          knnerr(dd)=min(sum((slicemat(1:total_train(dd),:)-descr{dd}).^2,2));
        end
        [appr_err,classifier(l)]=min(knnerr);
        erate = erate + ((classifier(l)) ~= lab);
        l=l+1;
        if(s==1)
           fprintf('%d...',l)
        end
    end
    fprintf('\n')
  end
  erate=erate/(l-1);

  fprintf('tested %d signals \n',l)

  out.erate=erate;

  else

  [res,KK]=size(moyenne{1});

  if fixed_dim ~=0
    if singlespace==1
      [alleigenvects,allsigmas]=PCA_basis_alltogether(descr,Dmax,moyenne);
    end
    for d=1:D
      if singlespace==0
         [eigenvects{d},sigmas{d}]=PCA_basis(descr{d},Dmax,moyenne{d});
      else
          eigenvects{d} = alleigenvects;
          sigmas{d}=allsigmas;
      end
      avevar(d) = mean(variance{d}(:));
      medvar(d) = median(variance{d}(:));
    end

    if maha>0
       LAM = mean(medvar) * 2 .^([-8:8]/4);
    else
      if fixed_dim == -1 
      % explore penalizations around the average variance of the training
        LAM = mean(avevar) * 2 .^([-8:8]/4);
      elseif fixed_dim == -2
        %hits=round(2*log2(Dmax));
        %LAM = floor(sqrt(2).^[0:hits]); 
        LAM = [0:2:Dmax];
      else
        LAM = fixed_dim;
      end
    end

    eratev=zeros(1,length(LAM));

    l=0;
    %%learning best lambda
    for d=1:D
            for s=pca_train(d)+1:total_train(d)
                    lab = d;
                    [slice]=scatt(train{d}{s},options)';
                    if maha>0
                    classifier = classif_mahalanobis(slice,moyenne, eigenvects, sigmas, LAM);
                    else
                    [classifier] = classif_optimized(slice, moyenne, eigenvects, LAM, fixed_dim,0);
                    end
                    eratev = eratev + (classifier ~= lab * ones(size(classifier)));
                    l=l+1;
            end
    end

    eratev=eratev/l;
    [best_p_err,best_lp] = min(eratev);

    fprintf('error rates on validation set are \n')
    100*best_p_err
    out.val_err = best_p_err;
    clear classifier;
    LLAM = LAM(best_lp);

  else
    LLAM=0;
    eigenvects=0;
  end


%%%%%%%%% END OF TRAINING %%%%%%%

%real testing
    erate=0;
    l=1;

    for d=1:D
      for s=1:size(test{d},2)		
        lab = d;
        [slice]=scatt(test{d}{s},options)';
        if maha>0
        [classifier(:,l),ncomps(:,l),approx_error(:,l),decomp] = classif_mahalanobis(slice,moyenne,eigenvects,sigmas,LLAM);
        else
        [classifier(:,l),ncomps(:,l),approx_error(:,l),decomp] = classif_optimized(slice, moyenne, eigenvects,LLAM,fixed_dim,0);
        end
        erate = erate + ((classifier(:,l)) ~= lab);
        ncomps_oracle(l) = ncomps(lab,l);
        rere=approx_error(:,l);
        [rsort,psort]=sort(rere,'ascend');
        secbest=psort(1);
        if secbest==lab
          secbest=psort(2);
        end
        approx_error_oracle(l) = approx_error(lab,l) / sum(slice(:).^2);
        approx_error_inter(l) = approx_error(secbest,l) / sum(slice(:).^2);
        l=l+1;
        if(s==1)
          fprintf('%d...',l)
        end
      end
      fprintf('\n')
    end

    erate=erate/(l-1);

    fprintf('tested %d signals \n',l)

    out.erate = erate;
    out.moyenne = moyenne;
    out.variance = variance;
    out.classifier = classifier;
    out.aprerr = mean(approx_error(:));
    if fixed_dim ~= 0
      out.eratev = eratev;
      out.best_lp = best_lp;
      out.eigenvects = eigenvects;
      out.sigmas = sigmas;
      out.ncomps = mean(ncomps(:));
      out.ncomps_or = mean(ncomps_oracle(:));
      out.avg_absorbed_ener_oracle = mean(approx_error_oracle(:));
      out.avg_absorbed_ener_row = mean(approx_error_inter(:));
      out.lambda = LLAM;
    end
end

end


